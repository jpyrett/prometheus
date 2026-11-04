"""Tests for the scoring and the decision rule of labs/module3_rag/other/compare_models.py (design choices entry 23).

Run:  pytest tests/test_compare_models.py -q

The comparison decides which model the Data Clerk uses, so a wrong scoring rule would give a wrong verdict without
any visible error. These tests feed the scorer made-up runs, with no model call and no network.
"""
import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from prometheus.rag.citations import CitationCheck

SPEC = importlib.util.spec_from_file_location("compare_models", Path(__file__).parents[1] / "labs" / "module3_rag" / "other" / "compare_models.py")
cm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cm)

BY_ID = {s.id: s for s in cm.SCENARIOS}


def run(answer="", calls=(("retrieve_news", {"tickers": ["MU"]}),), hits=(), bad_args=(), error=None, valid=(), invalid=()) -> dict:
    """A made-up run. `valid` are the citation numbers that point at a source, `invalid` those that do not."""
    check = CitationCheck(cited={n: object() for n in valid}, invalid=list(invalid), has_citations=bool(valid or invalid))
    return {"answer": answer, "calls": [{"name": n, "args": a} for n, a in calls], "hits": [{"text": t} for t in hits],
            "bad_args": list(bad_args), "error": error, "check": check}


# ---------------------------------------------------------------- answerable questions

def test_the_fact_reached_the_model_and_is_in_a_cited_answer_is_good():
    s = cm.score(BY_ID["mu-huang"], run("Huang says the shortage lasts until after 2029 [1].", hits=["... won't ease until after 2029 ..."], valid=[1]))
    assert s["good"] and s["found"] and not s["from_memory"]


def test_the_right_fact_without_a_citation_is_not_good():
    s = cm.score(BY_ID["mu-huang"], run("The shortage lasts until after 2029.", hits=["until after 2029"]))
    assert not s["good"] and s["found"]


def test_an_invalid_citation_is_not_good_and_is_counted():
    s = cm.score(BY_ID["mu-huang"], run("After 2029 [1][7].", hits=["until after 2029"], valid=[1], invalid=[7]))
    assert not s["good"] and s["invalid_citations"] == 1


def test_a_fact_that_never_reached_the_model_but_is_in_its_answer_came_from_memory():
    s = cm.score(BY_ID["mu-hbm-share"], run("Micron holds 18% of HBM [1].", hits=["some other article text"], valid=[1]))
    assert s["from_memory"] and not s["good"] and not s["found"]


def test_a_fact_that_never_reached_the_model_and_an_honest_answer_is_good_and_counted_as_an_honest_miss():
    s = cm.score(BY_ID["mu-hbm-share"], run("The sources do not say what HBM share Micron holds.", hits=["some other article text"]))
    assert s["good"] and s["honest_miss"] and not s["from_memory"]


def test_a_missing_fact_and_an_answer_that_neither_has_it_nor_admits_it_is_not_good():
    s = cm.score(BY_ID["mu-hbm-share"], run("Micron is a memory company.", hits=["some other article text"]))
    assert not s["good"] and not s["honest_miss"]


def test_the_fact_found_by_a_second_search_counts_because_every_search_of_the_run_is_pooled():
    s = cm.score(BY_ID["mu-hbm-share"], run("Micron holds 18% [2].", hits=["first search: nothing", "second search: only holds 18% hbm share"], valid=[2]))
    assert s["good"] and s["found"]


def test_the_answer_pattern_may_be_stricter_than_the_source_pattern():
    sc = BY_ID["mu-contracts"]  # the source only has to hold 35%, the answer must give 35% and 2030
    assert cm.score(sc, run("35% through 2030 [1].", hits=["secured 35% of revenue"], valid=[1]))["good"]
    assert not cm.score(sc, run("35% [1].", hits=["secured 35% of revenue"], valid=[1]))["good"]


def test_the_first_tool_must_be_retrieve_news_for_a_question_about_the_news():
    s = cm.score(BY_ID["mu-huang"], run("After 2029 [1].", calls=(("keyword_search", {}),), hits=["after 2029"], valid=[1]))
    assert not s["good"] and not s["first_tool_ok"]


def test_an_api_error_is_never_good():
    assert not cm.score(BY_ID["mu-huang"], run("After 2029 [1].", hits=["after 2029"], valid=[1], error="APIError"))["good"]


# ---------------------------------------------------------------- not in the news

@pytest.mark.parametrize("answer", ["The sources do not mention any partnership with Boeing.", "I found no information about that.",
                                    "Nothing in the retrieved news covers a Boeing deal.", "No supporting text was found.",
                                    "The stored news does not contain any such article."])
def test_an_honest_answer_to_an_unanswerable_question_is_good(answer):
    assert cm.score(BY_ID["mu-boeing"], run(answer, hits=["unrelated text"]))["good"]


def test_an_answer_that_makes_something_up_is_not_good():
    assert not cm.score(BY_ID["mu-boeing"], run("Micron and Boeing signed a memory supply deal [1].", hits=["unrelated text"], valid=[1]))["good"]


def test_an_unanswerable_question_still_needs_a_search_first():
    assert not cm.score(BY_ID["mu-boeing"], run("I have no information about that.", calls=()))["good"]


# ---------------------------------------------------------------- tool choice and no tool

def test_a_tool_question_needs_an_acceptable_first_tool_and_valid_arguments():
    sc = BY_ID["mu-financials"]
    assert cm.score(sc, run(calls=(("get_financials", {"symbol": "MU"}),)))["good"]
    assert not cm.score(sc, run(calls=(("retrieve_news", {}),)))["good"]
    assert not cm.score(sc, run(calls=(("get_financials", {"symbol": "../x"}),), bad_args=["get_financials: ToolInputError"]))["good"]
    assert cm.score(BY_ID["mu-trend"], run(calls=(("retrieve_news", {}),)))["good"]  # two tools are acceptable there


def test_no_tool_questions_are_good_only_without_a_tool_call_and_with_an_answer():
    sc = BY_ID["install"]
    assert cm.score(sc, run("I can't install software.", calls=()))["good"]
    assert not cm.score(sc, run("", calls=()))["good"]
    assert not cm.score(sc, run("Installing...", calls=(("fetch_filings", {}),)))["good"]


def test_a_greeting_that_explains_citing_has_no_invalid_citation_but_a_news_answer_that_cites_nothing_real_does():
    greeting = cm.score(BY_ID["hello"], run("I cite sources as [1], [2].", calls=(), invalid=[1, 2]))
    assert greeting["invalid_citations"] == 0 and greeting["good"]
    news = cm.score(BY_ID["mu-huang"], run("After 2029 [1][2].", hits=["after 2029"], valid=[1], invalid=[2]))
    assert news["invalid_citations"] == 1


def test_invented_tools_and_fetch_requests_are_counted():
    s = cm.score(BY_ID["mu-fresh"], run(calls=(("fetch_news", {"symbol": "MU"}), ("run_shell", {}), ("fetch_filings", {"symbol": "MU"}))))
    assert s["invented_tool"] and s["fetch_requests"] == 2


def test_the_ticker_filter_is_noticed_in_either_form():
    assert cm.score(BY_ID["mu-huang"], run(calls=(("retrieve_news", {"tickers": "MU"}),)))["used_ticker_filter"]
    assert cm.score(BY_ID["mu-huang"], run(calls=(("retrieve_news", {"tickers": ["NVDA", "MU"]}),)))["used_ticker_filter"]
    assert not cm.score(BY_ID["mu-huang"], run(calls=(("retrieve_news", {}),)))["used_ticker_filter"]


# ---------------------------------------------------------------- rate limits are not model failures

def test_a_provider_rate_limit_is_recognised_in_its_usual_forms():
    class Limit(Exception):
        status_code = 429

    class RateLimitError(Exception):
        pass

    assert cm.is_rate_limited(Limit("x"))
    assert cm.is_rate_limited(RateLimitError("x"))
    assert cm.is_rate_limited(Exception("Error code: 429 - {'error': {'message': 'Rate limit exceeded: free-models-per-day'}}"))
    assert cm.is_rate_limited(ValueError("Rate limit exceeded for this key"))
    assert not cm.is_rate_limited(ValueError("bad request: tools are not supported"))
    assert not cm.is_rate_limited(ConnectionError("could not reach the server"))


def test_the_error_hint_is_short_one_line_and_has_no_key():
    hint = cm.safe_hint(Exception("Error code: 401\n  with key sk-or-v1-abcdef1234567890 " + "x" * 400))
    assert "\n" not in hint and len(hint) <= 200 and "abcdef1234567890" not in hint and "sk-..." in hint


# ---------------------------------------------------------------- --agent: a Data Clerk result in the shape the scoring reads

def test_a_data_clerk_result_becomes_a_row_the_scoring_understands():
    from prometheus.agents.data_clerk.data_clerk_agent import Result
    result = Result(question="q", answer="After 2029 [1].", turns=2, tokens_in=10, tokens_out=5, has_citations=True,
                    references=[{"number": 1, "chunk_id": "a#0"}],
                    sources=[{"number": 1, "headline": "H", "score": 0.7, "text": "until after 2029", "chunk_id": "a#0"}],
                    tool_calls=[{"name": "retrieve_news", "args": {"tickers": ["MU"]}, "outcome": "ok"},
                                {"name": "get_financials", "args": {"symbol": "../x"}, "outcome": "invalid"}])
    row = cm.agent_row("m/x", BY_ID["mu-huang"], result, 1.234)
    assert row["calls"][0] == {"name": "retrieve_news", "args": {"tickers": ["MU"]}} and row["bad_args"] == ["get_financials: invalid arguments"]
    assert row["hits"][0]["text"] == "until after 2029" and row["seconds"] == 1.23 and row["error"] is None and not row["rate_limited"]
    s = cm.score(BY_ID["mu-huang"], row)
    assert s["found"] and s["cited"] and s["good"]  # the answer has the fact, the source has it, the citation is valid, the first tool is right


def test_an_agent_model_failure_and_a_rate_limit_carry_over_to_the_row():
    from prometheus.agents.data_clerk.data_clerk_agent import Result
    failed = Result(question="q", error="RuntimeError", rate_limited=True, problems=["the model call failed (RuntimeError)"])
    row = cm.agent_row("m/x", BY_ID["mu-huang"], failed, 0.5)
    assert row["error"] == "RuntimeError" and row["rate_limited"] and "model call failed" in row["error_hint"]
    assert cm.agent_row("m/x", BY_ID["mu-huang"], Result(question="q"), 0.1)["error"] is None


def test_the_unavailable_toolbox_refuses_every_tool_so_nothing_can_be_fetched():
    box = cm.Unavailable()
    for call in (lambda: box.fetch_news(symbol="MU"), lambda: box.search("MU", ["x"]), lambda: box.discover("x"), lambda: box.performance_summary("MU"),
                 lambda: box.financials.get("MU")):
        with pytest.raises(RuntimeError, match="not available in this comparison"):
            call()


# ---------------------------------------------------------------- the admission phrases must not favour one model's wording

@pytest.mark.parametrize("answer", [
    "The stored news gives SK Hynix's HBM share but not Micron's, so a direct comparison isn't possible.",   # DeepSeek, mu-hbm-share
    "I found no stored news about a Micron antitrust lawsuit or any ruling in one.",                         # DeepSeek, mu-antitrust
    "The sources give the percentage but do not say through which year those contracts run.",                # DeepSeek, mu-contracts
    "The stored news does not contain any information about a partnership.",                                 # gpt-oss
    "I couldn\u2019t find any stored news covering that.",                                                    # gpt-oss, curly apostrophe
    "None of the retrieved articles mention Boeing."])
def test_the_ways_the_real_models_say_the_sources_do_not_cover_it_are_recognised(answer):
    assert cm.admits(answer)


@pytest.mark.parametrize("answer", ["Micron and Boeing signed a memory supply deal [1].", "Micron holds 18% of HBM [2].",
                                    "Nvidia is standardizing NVHBM with Micron and SanDisk [1]."])
def test_a_confident_answer_is_not_an_admission(answer):
    assert not cm.admits(answer)


# ---------------------------------------------------------------- the informational metrics

def test_numbers_in_an_answer_that_no_source_has_are_found():
    sources = "revenue grew 256% to $54.23 billion, with 35% of revenue secured through 2030"
    assert cm.unsupported_numbers("Revenue grew 256% [1] and 35% is secured through 2030.", sources, "q") == []
    assert cm.unsupported_numbers("Revenue grew 300% to $60 billion.", sources, "q") == ["300%", "60b"]
    assert cm.unsupported_numbers("Revenue was $54.23 million.", sources, "q") == ["54.23m"]  # the unit matters: billion is not million


def test_numbers_that_are_only_citation_marks_counts_or_in_the_question_are_not_flagged():
    assert cm.unsupported_numbers("Five articles say so [1][2]【3】 and 3 of them agree.", "no numbers here", "q") == []
    assert cm.unsupported_numbers("It was 2026.", "x", "What happened in 2026?") == []
    # the guard matters for numbers over 99, which the lone-integer rule would not hide: a source number or a long marker
    assert cm.unsupported_numbers("Sources [123] and 【456†L10-L20】 agree.", "no numbers here", "q") == []


def test_a_figure_written_differently_is_still_matched():
    assert cm.unsupported_numbers("It reached $1,000 million.", "reached $1,000 million", "q") == []
    assert cm.unsupported_numbers("About 54.23 billion.", "revenue of $54.23 billion", "q") == []


def test_fetches_on_a_news_question_are_counted_apart_from_the_expected_ones():
    on_news = cm.score(BY_ID["mu-margin"], run("x", calls=(("retrieve_news", {}), ("fetch_news", {}), ("fetch_news", {}))))
    expected = cm.score(BY_ID["mu-fresh"], run("x", calls=(("fetch_news", {"symbol": "MU"}),)))
    assert on_news["fetch_on_news_question"] == 2 and on_news["fetch_requests"] == 2
    assert expected["fetch_on_news_question"] == 0 and expected["fetch_requests"] == 1


# ---------------------------------------------------------------- the decision rule

def summary(good=0.80, memory=0, invalid=0, invented=0, fetches=0, cost=0.001, seconds=10.0) -> dict:
    return {"good": good, "from_memory": memory, "invalid_citations": invalid, "invented_tools": invented, "fetch_requests": fetches,
            "cost_per_run": cost, "seconds": seconds}


def verdict_of(candidate: dict, current: dict | None = None) -> str:
    out = io.StringIO()
    with redirect_stdout(out):
        cm.verdict(["cur/a", "cand/b"], {"cur/a": current or summary(), "cand/b": candidate})
    return out.getvalue().splitlines()[-1]


def test_a_clearly_better_and_no_worse_candidate_wins():
    assert verdict_of(summary(good=0.90)) == "VERDICT: SWITCH"


def test_a_gain_under_five_points_is_not_enough():
    assert verdict_of(summary(good=0.84)).startswith("VERDICT: KEEP")
    assert verdict_of(summary(good=0.85)).startswith("VERDICT: SWITCH")  # exactly 5 points is enough


@pytest.mark.parametrize("worse", [dict(memory=1), dict(invalid=1), dict(invented=1), dict(fetches=1)])
def test_any_safety_regression_blocks_the_switch(worse):
    assert verdict_of(summary(good=0.95, **worse)).startswith("VERDICT: KEEP")


def test_an_equal_safety_record_is_not_a_regression():
    assert verdict_of(summary(good=0.95, memory=2, fetches=3), summary(memory=2, fetches=3)).startswith("VERDICT: SWITCH")


def test_cost_and_time_limits():
    assert verdict_of(summary(good=0.95, cost=0.00125)).startswith("VERDICT: SWITCH")  # exactly 1.25x
    assert verdict_of(summary(good=0.95, cost=0.00130)).startswith("VERDICT: KEEP")
    assert verdict_of(summary(good=0.95, seconds=15.0)).startswith("VERDICT: SWITCH")  # exactly 1.5x
    assert verdict_of(summary(good=0.95, seconds=15.5)).startswith("VERDICT: KEEP")


def test_when_the_cost_is_unknown_the_rule_is_decided_on_the_other_criteria():
    assert verdict_of(summary(good=0.95, cost=None), summary(cost=None)).startswith("VERDICT: SWITCH")
