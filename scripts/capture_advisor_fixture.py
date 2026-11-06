"""Run the Advisor's Generate Recommendations step on REAL data for one symbol and save what went in, so that a test can replay it offline (Advisor v1, design/week5_advisor_implementation.md).

    python scripts/capture_advisor_fixture.py --symbol SNDK --sector XLK --company Sandisk            # says what it would do and stops
    python scripts/capture_advisor_fixture.py --symbol SNDK --sector XLK --company Sandisk --yes       # does it

WHAT IT CALLS (so it asks for --yes): the Data Clerk's model (one answer for the work order, OpenRouter, paid), Finnhub's metric endpoint for every candidate (paid), the Advisor's model
(the plan call and the thesis call, OpenRouter, paid). It reads the stored news, press releases, filings and daily bars and fetches NOTHING itself (run `scripts/fetch_bars.py` and the Data
Clerk's fetch tools first: this is the step after the refresh). It writes only `tests/fixtures/advisor_real/<SYMBOL>.json` and a ledger in a temporary folder: not data/ledger, not data/processed.

WHAT IS SAVED: the work order the Data Clerk produced (the real sources and the real filing evidence), the metrics the Data Clerk calculated for every candidate (the real numbers, with
their sources and notes), and every raw reply of the Advisor's model (the plan call and the thesis call, with a retry of either when there was one). `tests/test_advisor_sndk_pg.py` feeds them back through the real Advisor code with a scripted model that returns the
recorded replies: the same code, the same numbers, the same words, no network.
"""
import argparse
import json
import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage  # noqa: E402

from prometheus.agents.advisor.advisor import Advisor  # noqa: E402
from prometheus.agents.data_clerk.data_clerk_agent import DataClerk  # noqa: E402
from prometheus.common import work_order as wo  # noqa: E402
from prometheus.common.portfolio import load_portfolio  # noqa: E402
from prometheus.memory.ledger import Ledger  # noqa: E402
from prometheus.models.chat import get_chat_model  # noqa: E402
from prometheus.rag.config import REPO_ROOT  # noqa: E402

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "advisor_real"


class RecordingModel:
    """The real chat model, remembering the text of every reply."""

    def __init__(self, model):
        self.model, self.replies, self.model_name = model, [], getattr(model, "model_name", None)

    def invoke(self, messages):
        reply = self.model.invoke(messages)
        self.replies.append(reply.content if isinstance(reply.content, str) else "")
        return AIMessage(content=reply.content, usage_metadata=getattr(reply, "usage_metadata", None))


class RecordingClerk:
    """The real Data Clerk, remembering the metrics it calculated."""

    def __init__(self, clerk):
        self.clerk, self.tools, self.metrics = clerk, clerk.tools, {}

    def request_metrics(self, order, today=None):
        out = self.clerk.request_metrics(order, today)
        if out.data:
            self.metrics = out.data.metrics
        return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--company", required=True, help="the company's name, for the question the Data Clerk answers")
    parser.add_argument("--sector", help="the sector ETF of the symbol (XLK, XLP, ...), as a user would give it")
    parser.add_argument("--today", default=date.today().isoformat(), help="the run date (YYYY-MM-DD)")
    parser.add_argument("--portfolio", default="configs/portfolios/jacob_pyrett.json")
    parser.add_argument("--since-days", type=int, default=14)
    parser.add_argument("--name", help="the file name stem (default: the symbol): a second capture of the same symbol, to keep two runs of a borderline one")
    parser.add_argument("--yes", action="store_true", help="really run (paid APIs)")
    args = parser.parse_args(argv)
    symbol, today = args.symbol.strip().upper(), date.fromisoformat(args.today)
    print(f"{symbol} ({args.company}), sector {args.sector}, as of {today}, portfolio {args.portfolio} -> {FIXTURES / ((args.name or symbol) + '.json')}")
    if not args.yes:
        print("not run: add --yes (it calls the Data Clerk's model, Finnhub and the Advisor's model: paid APIs)")
        return 0
    portfolio = load_portfolio(REPO_ROOT / args.portfolio)
    order = wo.new_work_order(f"What do the stored news, press releases and filings say about {args.company} ({symbol})?", subject=[symbol], since_days=args.since_days, top_k=10,
                              sector=args.sector, case="funds")
    DataClerk(allow_fetch=False).request_data(order, today)
    if order.status != "data_ready":
        print("the Data Clerk's order is not ready:", order.errors)
        return 1
    before = order.model_dump(mode="json")
    model, clerk = RecordingModel(get_chat_model("advisor")), RecordingClerk(DataClerk(model=object()))
    with tempfile.TemporaryDirectory() as tmp:
        Advisor(model=model, clerk=clerk, ledger=Ledger(Path(tmp) / "ledger.jsonl")).generate_recommendations(order, portfolio, "funds", today)
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / f"{args.name or symbol}.json").write_text(json.dumps({
        "symbol": symbol, "company": args.company, "sector": args.sector, "today": today.isoformat(), "portfolio": args.portfolio, "order": before, "metrics": clerk.metrics,
        "replies": model.replies}, indent=1, default=str) + "\n", encoding="utf-8")   # every reply of the Advisor's model, in the order it was asked: the plan call (and its retry), then the thesis call (and its retry)
    print("model calls:", [(r["stage"], r["calls"], r["error"]) for r in order.advisor_runs], "| thesis:", [(r["symbol"], (r["thesis"]["thesis_error"] or "written")[:100]) for r in order.recommendations])
    print(f"status {order.status}; recommendations {[(r['symbol'], r['action']['action'], r['score']) for r in order.recommendations]}; "
          f"candidates {[(c['symbol'], c['reason'][:60]) for c in order.candidates]}; errors {order.errors}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
