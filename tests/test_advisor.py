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
from prometheus.agents.data_clerk.data_clerk_agent import Result
from prometheus.common import work_order as wo
from prometheus.prompts import advisor as prompts

HUANG = ("Memory Stocks Are Plummetting. Jensen Huang Says the Bears Have It Wrong.\n\nMemory chip stocks have declined roughly one-third "
         "from their July peak amid investor concerns about cyclical oversupply and rising debt yields. However, Nvidia CEO Jensen Huang "
         "argues the shortage won't ease until after 2029, and memory stocks trade at attractive valuations with extreme profit margins.")
CONTRACTS = ("Micron Has Locked In Its Revenue\n\nMicron has secured 35% of its revenue through long-term contracts, which reduces the "
             "cyclical risk of its memory business. Sandisk has secured 50%.")


def source(number, text, tickers=("MU",), headline=None):
    headline = headline or f"Headline number {number}"  # distinct, because the same headline means the same story
    return {"number": number, "chunk_id": f"art{number}#0", "article_id": f"art{number}", "chunk": 1, "chunks": 1, "headline": headline,
            "publisher": "The Motley Fool", "published_utc": "2026-10-23T10:00:00Z", "url": f"https://example.com/{number}",
            "tickers": list(tickers), "text": text, "score": 0.6}


def ready_order(*sources):
    order = wo.new_work_order("AI memory shortage. Which companies are affected, and what does the recent news say?", since_days=7, top_k=5)
    cited = [{"number": 1, "chunk_id": sources[0]["chunk_id"]}] if sources else []
    return wo.attach_result(order, Result(question="q", answer="Yes [1]." if sources else "Nothing found.", references=cited, sources=list(sources),
                                          no_supporting_text=not sources, tool_calls=[{"name": "retrieve_news", "args": {}, "outcome": "ok"}]))


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


# real text from the Procter & Gamble run (news/PG.txt, 2026-10-23 and 24), after the loader decodes `&amp;`
PG_FREEDOM = "Freedom Broker Upgrades Procter & Gamble to Buy from Hold\n\nFreedom Broker upgraded Procter & Gamble from Hold to Buy."
PG_EARNINGS = ("Procter & Gamble's Earnings Show a Resilient Consumer Despite Price Hikes. Is the Dividend King Stock a Buy?\n\n"
               "Procter & Gamble reported strong Q1 fiscal results with 4% organic sales growth and beat earnings estimates, causing the stock to jump 5.6%.\n\n"
               "PG | positive | P&G delivered surprisingly strong Q1 results with 4% organic sales growth, beat revenue and EPS estimates, and demonstrated "
               "resilience despite inflationary pressures. The stock had its best day in over six years with a 5.6% jump.")
PG_CITI = ("Citigroup Maintains Buy on Procter & Gamble, Raises Price Target to $187\n\n"
           "Citigroup  analyst Filippo Falorni   maintains Procter & Gamble (NYSE:PG) with a Buy and raises the price target from $180 to $187.")
PG_SIGNALS = [sig(1, "bullish", "Freedom Broker Upgrades Procter & Gamble to Buy from Hold"),
              sig(2, "bullish", "P&G delivered surprisingly strong Q1 results with 4% organic sales growth, beat revenue and EPS estimates, and demonstrated resilience despite inflationary pressures."),
              sig(3, "bullish", "Citigroup analyst Filippo Falorni maintains Procter & Gamble (NYSE:PG) with a Buy and raises the price target from $180 to $187.")]


def pg_order(company="Procter & Gamble Co"):
    sources = [source(1, PG_FREEDOM, ("PG",), "Freedom Broker Upgrades Procter & Gamble to Buy from Hold"),
               source(2, PG_EARNINGS, ("PG",), "Procter & Gamble's Earnings Show a Resilient Consumer Despite Price Hikes. Is the Dividend King Stock a Buy?"),
               source(3, PG_CITI, ("PG",), "Citigroup Maintains Buy on Procter & Gamble, Raises Price Target to $187")]
    order = wo.new_work_order("What is the latest news on PG?", subject=["PG"], company=company, since_days=7, top_k=5)
    return wo.attach_result(order, Result(question="q", answer="Yes [1].", references=[{"number": 1, "chunk_id": sources[0]["chunk_id"]}], sources=sources,
                                          no_supporting_text=False, tool_calls=[{"name": "retrieve_news", "args": {}, "outcome": "ok"}]))


def test_the_work_orders_company_name_gives_the_short_name_even_when_the_model_writes_only_the_ticker_as_the_company():
    order = pg_order("Procter & Gamble Co")
    adv.Advisor(Scripted(reply(rec("PG", "PG", "bullish", 70, [PG_SIGNALS[1]])))).advise(order)
    assert order.recommendations[0]["signals"][0]["verified"] is True
    other = pg_order(company=None)  # no name on the order and none from the model that makes "P&G"
    adv.Advisor(Scripted(reply(rec("PG", "PG", "bullish", 70, [PG_SIGNALS[1]])))).advise(other)
    assert other.recommendations[0]["signals"][0]["reason"] == "the quote does not name the company"


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


@pytest.mark.parametrize("text, problem", [("", "no JSON object found"), ("no json here", "no JSON object found"), ("{not json}", "not valid JSON"),
                                            ('{"other": 1}', "no recommendations list"), ('{"recommendations": "x"}', "no recommendations list")])
def test_a_reply_without_a_recommendations_list_is_refused_with_the_reason(text, problem):
    with pytest.raises(ValueError, match=problem):
        adv.parse_reply(text)


# ---------------------------------------------------------------- one recommendation


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


# ---------------------------------------------------------------- the Advisor


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


def test_a_model_failure_is_recorded_and_flagged_when_it_is_a_rate_limit_and_leaks_nothing():
    order = ready_order(source(1, CONTRACTS))
    adv.Advisor(model=Scripted(RuntimeError("Error code: 429 - Rate limit exceeded sk-or-v1-SECRET"))).advise(order)
    assert order.status == "data_ready" and order.errors == ["the advisor model call failed (RuntimeError)"] and order.advisor_runs[0]["rate_limited"] is True
    assert "SECRET" not in order.model_dump_json()


def test_more_than_five_companies_are_cut_and_the_cut_is_noted():
    order = ready_order(source(1, CONTRACTS))
    adv.Advisor(model=Scripted(reply(*[rec(symbol=s, company=s) for s in ("MU", "NVDA", "AMD", "INTC", "TSM", "ASML", "QCOM")]))).advise(order)
    assert len(order.recommendations) == 5 and any("2 recommendation(s) beyond the first 5" in d for d in order.advisor_runs[0]["dropped"])


# ---------------------------------------------------------------- the prompt


# ================================================================ the Advisor reads data.results and data.filing_evidence (design choices entry 40)
#
# The evidence is the REAL one: what `recommendation_evidence` makes of the real META filings of tests/fixtures/filings_meta_sample.txt (no balance sheet: the 10-Q has no text there).
# E2 and E3 are negative-type (an officer's base salary, restricted stock units), E6 and E7 positive-type (dividend passages of the 8-K about the recapitalization).

from pathlib import Path  # noqa: E402

from prometheus.agents.data_clerk import filing_signals  # noqa: E402
from prometheus.rag.loaders import load_filings  # noqa: E402

META_FILINGS = Path(__file__).parent / "fixtures" / "filings_meta_sample.txt"


def meta_evidence() -> dict:
    return filing_signals.recommendation_evidence("META", filing_signals.filings_from_documents(load_filings(META_FILINGS)), today=date(2026, 11, 1))


def meta_source(number, text, headline=None):
    return {**source(number, text, tickers=("META",), headline=headline)}


META_NEWS = "Meta Platforms has told investors it will keep spending heavily on AI chips this year."


def meta_order_with_evidence(sources=(), results=None):
    order = wo.new_work_order("What do Meta's filings say?", subject=["META"], since_days=365, source="filings", action_type="search_filing")
    sources = list(sources) or [meta_source(1, META_NEWS)]
    r = Result(question="q", answer="Yes [1].", references=[{"number": 1, "chunk_id": sources[0]["chunk_id"]}], sources=sources,
               tool_calls=[{"name": "retrieve_filings", "args": {}, "outcome": "ok"}], filing_evidence={"META": meta_evidence()})
    wo.attach_result(order, r)
    return order


def evidence_quote(item_id, limit=60):
    return next(e for e in meta_evidence()["evidence"] if e["id"] == item_id)["quote"][:limit]


def test_the_advisor_is_told_which_sources_are_news_press_releases_and_filings_from_the_results():
    sources = [meta_source(1, META_NEWS), {**meta_source(2, "Meta announced a dividend."), "chunk_id": "pr:x#0"}, {**meta_source(3, "Item 5.02 Meta."), "chunk_id": "sec:0000000001-25-000001:items#0", "swept": True}]
    order = meta_order_with_evidence(sources)
    text = prompts.kinds_text(order)
    assert "News (independent reporting): [1]; the Data Clerk's answer cites [1]" in text
    assert "Press releases (the company's own announcements, not independent): [2]" in text and "cites [2]" not in text
    assert "SEC filings (the company's own statements to the regulator, not independent): [3] (found afterwards by code)" in text
    assert prompts.kinds_text(wo.new_work_order("q")) is None and "- SEC filings (the company's own statements to the regulator, not independent): none" in prompts.kinds_text(ready_order(source(1, CONTRACTS)))


@pytest.mark.parametrize("signal, reason", [
    ({"source": "E2", "direction": "bullish", "quote": "$QUOTE2"}, "the direction contradicts the evidence: a negative-type item (executive_compensation) is bearish"),
    ({"source": "E6", "direction": "bearish", "quote": "$QUOTE6"}, "the direction contradicts the evidence: a positive-type item (dividend) is bullish"),
    ({"source": "E2", "direction": "bearish", "quote": "Powell McCormick will receive a base salary of ten million dollars"}, "the quote is not in evidence E2"),
    ({"source": "E2", "direction": "bearish", "quote": "base salary"}, "the quote is missing or too short to check"),
    ({"source": "E99", "direction": "bearish", "quote": "$QUOTE2"}, "evidence 'E99' is not one of this company's evidence items"),
    ({"source": "E2", "direction": "sideways", "quote": "$QUOTE2"}, "the direction must be bullish or bearish"),
])
def test_an_evidence_signal_that_breaks_a_rule_is_kept_with_its_reason_and_does_not_count(signal, reason):
    order = meta_order_with_evidence()
    signal = {**signal, "quote": signal["quote"].replace("$QUOTE2", evidence_quote("E2")).replace("$QUOTE6", evidence_quote("E6"))}
    adv.Advisor(model=Scripted(reply(rec("META", "Meta Platforms", signals=[signal])))).advise(order)
    r = order.recommendations[0]
    assert r["signals"][0]["verified"] is False and r["signals"][0]["reason"] == reason and r["score"] is None and r["signals_counted"] == 0


def test_five_passages_on_one_topic_are_one_voice_and_the_sources_and_the_evidence_have_a_cap_each():
    order = meta_order_with_evidence()
    same_topic = [{"source": n, "direction": "bullish", "quote": evidence_quote(n)} for n in ("E6", "E7")]
    adv.Advisor(model=Scripted(reply(rec("META", "Meta Platforms", signals=same_topic)))).advise(order)
    first, second = order.recommendations[0]["signals"]
    assert first["counted"] is True and second["verified"] is True and second["counted"] is False and second["note"].startswith("not counted: the same topic")
    assert order.recommendations[0]["bullish_signals"] == 1
    many = [sig(1, "bullish", "Meta Platforms has told investors it will keep spending heavily")] * 1 + [{"source": f"E{n}", "direction": "bearish" if n < 5 else "bullish", "quote": evidence_quote(f"E{n}")} for n in (1, 2, 3, 4, 5, 6)]
    again = meta_order_with_evidence()
    adv.Advisor(model=Scripted(reply(rec("META", "Meta Platforms", signals=many)))).advise(again)
    kinds = [bool(s.get("evidence_id")) for s in again.recommendations[0]["signals"]]
    assert kinds.count(True) == 3 and kinds.count(False) == 1  # 3 from the evidence at most, apart from the 3 from the sources


