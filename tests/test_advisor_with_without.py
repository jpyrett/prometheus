"""Tests for the numbers that labs/module3_rag/advisor_with_without.py reports (Part 8 of the RAG guide).

Run:  pytest tests/test_advisor_with_without.py -q

The script prints the table that the guide and the week 3 submission quote, so the arithmetic behind it is tested here with hand-built
orders: no model, no network, no cost.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

from prometheus.common import work_order as wo

SCRIPT = Path(__file__).resolve().parents[1] / "labs" / "module3_rag" / "advisor_with_without.py"
spec = importlib.util.spec_from_file_location("advisor_with_without", SCRIPT)
awo = importlib.util.module_from_spec(spec)
sys.modules["advisor_with_without"] = awo
spec.loader.exec_module(awo)


def signal(verified=True, reason=None, direction="bullish", source=1):
    return {"source": source, "chunk_id": "a#0" if verified else None, "direction": direction, "quote": "q", "verified": verified, "reason": reason}


def rec(symbol, model_score, score, stance="bullish", signals=()):
    consistent = None if score is None else (stance == ("bullish" if score > 50 else "bearish" if score < 50 else "neutral"))
    return {"symbol": symbol, "model_score": model_score, "score": score, "supported": score is not None, "stance": stance, "consistent": consistent,
            "signals": list(signals)}


def order(*recs, tokens=(100, 10), status="recommended", errors=()):
    o = wo.new_work_order("q")
    o.status, o.recommendations, o.errors = status, list(recs), list(errors)
    o.advisor_runs = [{"tokens_in": tokens[0], "tokens_out": tokens[1]}]
    return o


def test_spread_is_the_range_and_needs_two_values():
    assert awo.spread([55, 78, 70]) == 23 and awo.spread([60, 60]) == 0 and awo.spread([5]) is None and awo.spread([]) is None


def test_the_summary_counts_signals_and_says_why_the_failures_failed():
    s = awo.summarize([order(rec("MU", 70, 67, signals=[signal(), signal(False, "the quote is not in source 1"), signal(False, "the quote does not name the company")]),
                             rec("NVDA", 60, None, signals=[signal(False, "source 9 is not one of the sources shown"), signal(False, "source 4 is not one of the sources shown")]))])
    assert (s["signals_claimed"], s["signals_verified"]) == (5, 1) and (s["recommendations"], s["with_calculated_score"]) == (2, 1)
    assert s["signal_failures"] == {"the quote is not in source 1": 1, "the quote does not name the company": 1, "source N is not one of the sources shown": 2}


def test_the_summary_compares_the_companies_between_runs():
    runs = [order(rec("MU", 70, 67), rec("NVDA", 60, 60)), order(rec("MU", 55, 67), rec("AMD", 50, 60)), order(rec("MU", 78, 67))]
    s = awo.summarize(runs)
    assert s["runs"] == 3 and s["recommendations_per_run"] == 1.7 and s["companies_in_any_run"] == 3 and s["companies_in_every_run"] == ["MU"]


def test_the_summary_measures_how_much_each_score_moves_between_runs():
    runs = [order(rec("MU", 70, 67), rec("NVDA", 65, 60)), order(rec("MU", 55, 67), rec("NVDA", 75, 60)), order(rec("MU", 78, 67))]
    s = awo.summarize(runs)
    # MU: model 55..78 = 23, calculated 0. NVDA (two runs): model 65..75 = 10, calculated 0. The median of 23 and 10 is 16.5.
    assert s["model_score_spread_median"] == 16.5 and s["calculated_score_spread_median"] == 0.0


def test_a_company_in_only_one_run_has_no_spread_and_a_score_that_is_none_does_not_count_as_one():
    s = awo.summarize([order(rec("MU", 70, None)), order(rec("NVDA", 60, 60))])
    assert s["model_score_spread_median"] is None and s["calculated_score_spread_median"] is None and s["with_calculated_score"] == 1


def test_the_models_own_confidence_is_summarized_from_its_own_numbers():
    s = awo.summarize([order(rec("A", 82, None), rec("B", 78, None), rec("C", 38, None, stance="bearish"), rec("D", None, None))])
    assert s["model_score_mean"] == 66.0 and s["model_score_70_or_more"] == 2 and s["model_scores"] == 3  # D has no number and is not counted


def test_the_stance_is_compared_with_the_calculated_score_only_where_there_is_one():
    s = awo.summarize([order(rec("A", 70, 67, "bullish"), rec("B", 70, 33, "bullish"), rec("C", 70, None, "bullish"))])
    assert s["stance_agrees_with_score"] == "1 of 2"


def test_tokens_add_up_across_runs_and_failed_runs_are_counted_with_their_errors():
    s = awo.summarize([order(rec("MU", 70, 67), tokens=(500, 100)), order(tokens=(300, 20), status="data_ready", errors=["the advisor model call failed (X)"])])
    assert (s["tokens_in"], s["tokens_out"], s["failed_runs"]) == (800, 120, 1) and s["advisor_errors"] == [["the advisor model call failed (X)"]]


def test_an_empty_condition_summarizes_without_dividing_by_zero():
    s = awo.summarize([order(), order()])
    assert s["recommendations_per_run"] == 0 and s["companies_in_any_run"] == 0 and s["model_score_mean"] is None and s["companies_in_every_run"] == []
    assert awo.summarize([])["runs"] == 0


def test_the_unavailable_toolbox_refuses_every_tool():
    box = awo.Unavailable()
    for call in (lambda: box.fetch_news(symbol="MU"), lambda: box.search("MU", ["x"]), lambda: box.financials.get("MU")):
        with pytest.raises(RuntimeError, match="not available in this comparison"):
            call()
