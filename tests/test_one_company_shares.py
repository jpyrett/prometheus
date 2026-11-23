"""ONE COMPANY, AND A NUMBER OF SHARES WITH NO STOCK NAMED (design choices entry 146; Jacob, 2026-11-22: "buy 5 shares of the most recommended shoe company" spent all $90,966.02 on 473 CROX, 477 UAA and 477 NKE).

REAL DATA ONLY: the model's real replies (`tests/fixtures/session_real/request_numbers_forms.json`, `scripts/capture_request_numbers_forms.py`) and a real Auto session of the whole desk on a throwaway account
(`tests/fixtures/session_real/auto_one_company_five_shares.json`, `python -m prometheus.application.session ... --save`, re-recorded for entry 147). The model read the request with no count and no shares; code now reads both from the words.
And "the most recommended company" is the Advisor's recommendation or nothing (entry 147): a neutral company never takes the place of the one the Critic denied."""
import json
from pathlib import Path

import pytest

from prometheus.agents.account_manager import intent as it

FIXTURES = Path(__file__).parent / "fixtures" / "session_real"
FORMS = json.loads((FIXTURES / "request_numbers_forms.json").read_text(encoding="utf-8"))["requests"]
SHOE = "buy 5 shares of the most recommended shoe company"


def read(typed: str) -> it.TradeIntent:
    return it.parse_intent(FORMS[typed]["replies"][-1], typed)


def test_jacobs_request_is_five_shares_of_one_company():
    raw = json.loads(FORMS[SHOE]["replies"][-1][FORMS[SHOE]["replies"][-1].find("{"):FORMS[SHOE]["replies"][-1].rfind("}") + 1])
    assert raw.get("count") is None      # the model's own reading has no count: the default of 3 would have been used
    intent = read(SHOE)
    assert (intent.action, intent.count, intent.shares_each, intent.budget, intent.said_how_much()) == ("buy", 1, 5, None, True)
    assert intent.describe().startswith("buy the most recommended 'shoe' company; if approved, 5 shares of it")


def test_the_singular_is_one_company_and_the_plural_is_not():
    assert read("buy the best AI stock").count == 1
    look = read("Recommend the best oil company")
    assert (look.action, look.count) == ("research", 1) and "bought" not in look.corrected[-1]
    assert not it.one_stock_asked("buy the best shoe companies") and not it.one_stock_asked("show me the top stock picks") and not it.one_stock_asked("buy the top 3 shoe stocks")


def test_shares_for_several_companies_without_each_stop_and_ask():
    with pytest.raises(it.NeedsClarification, match="5 shares but not which stock"):
        read("buy 5 shares of the top 3 shoe stocks")
    assert read("buy 5 shares of each of the top 3 oil stocks if recommended").shares_each == 5      # "each" says it: unchanged (entry 140)


def test_the_most_recommended_company_is_the_advisors_pick_or_nothing():
    """Entry 147 (Jacob: "The Recommended company was VFC and the wrong company CROX came back instead"): a real Auto session of the whole desk on a throwaway account with $90,966.02. The Advisor recommended ONE company and the
    Critic denied it (the weekend: market_closed, record_expired): nothing is bought, and no neutral company takes its place, each told why."""
    capture = json.loads((FIXTURES / "auto_one_company_five_shares.json").read_text(encoding="utf-8"))
    assert capture["intent"]["count"] == 1 and capture["intent"]["shares_each"] == 5
    order = capture["order"]
    [pick] = order["recommendations"]
    [review] = order["reviews"]
    assert review["approved"] is False and pick["symbol"] == "CROX"
    assert [(d["symbol"], d["will_trade"]) for d in order["trade_decisions"]] == [("CROX", "no_system_denied")]      # the pick's own decision, and NO neutral decision
    assert {"VFC", "NKE", "UAA"} <= {c["symbol"] for c in order["candidates"] if c.get("stance") == "neutral"}      # neutral companies were there to fill the place, and none was given a decision
    assert order["trades"] == [] and "trader" not in capture["path"]
