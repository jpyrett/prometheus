"""The work order and the status lines list the number of companies the request asked for (design choices entry 144). Real capture: "Tell me the top 5 recommended stocks in S&P 500."
(`tests/fixtures/session_real/sp500_top5.json`, a saved live session): the Data Clerk found 10 companies, the Advisor scored all 10 and recommended 2 (JNJ, AAPL); the other 8 were neutral."""
import json
from pathlib import Path
from types import SimpleNamespace

from prometheus.agents.account_manager.intent import TradeIntent
from prometheus.application import session
from prometheus.application.web import views
from prometheus.common.work_order import WorkOrder

CAPTURE = json.loads((Path(__file__).parent / "fixtures" / "session_real" / "sp500_top5.json").read_text())


def test_the_page_lists_the_top_5_and_names_the_rest():
    view = views.work_order_view(CAPTURE["order"], top=CAPTURE["intent"]["count"])
    # the # is the overall rank, 1 to 5, in that order (entry 145): JNJ and AAPL were the Advisor's two recommendations and 3rd and 4th of the ten it scored
    assert [(r["rank"], r["symbol"], r["group"]) for r in view["rows"]] == [(1, "PG", "scored"), (2, "BRK.A", "scored"), (3, "JNJ", "recommended"), (4, "AAPL", "recommended"), (5, "AMZN", "scored")]
    assert [r["detail"]["header"]["rank"] for r in view["rows"]] == [1, 2, 3, 4, 5]
    assert view["beyond"] == {"top": 5, "count": 5, "symbols": ["MSFT", "GOOGL", "NVDA", "UNH", "TSLA"]}
    every = views.work_order_view(CAPTURE["order"])      # no number (a Time Machine day): every row, nothing beyond
    assert [r["rank"] for r in every["rows"]] == list(range(1, 11)) and every["beyond"] is None


def test_the_status_lines_stop_at_the_number_asked_for():
    order = WorkOrder.model_validate(CAPTURE["order"])
    companies = [{"symbol": s, "company": None} for s in order.subject]
    lines = session.status_lines(SimpleNamespace(order=order, companies=companies, notes=[]), TradeIntent.model_validate(CAPTURE["intent"]), 5)
    text = "\n".join(lines)
    assert "asked for the top 5" in text and "10 listed companies to score" in text and "10 listed\n" not in text
    assert "2 of the top 5 passed a strategy" in text
    numbered = [line.split()[:2] for line in lines if line.startswith("  #")]
    assert numbered == [["#1", "PG"], ["#2", "BRK.A"], ["#3", "JNJ"], ["#4", "AAPL"], ["#5", "AMZN"]]      # five items, numbered 1 to 5 by the overall score (entry 145)
    assert "5 more scored below your top 5, not shown" in text
