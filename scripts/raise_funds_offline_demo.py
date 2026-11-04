"""RaiseFunds OFFLINE demo: the whole Account Manager flow with NO paid call (design choices entry 42).

    python scripts/raise_funds_offline_demo.py                  # buy 2 MU, with the sample portfolio of Jacob Pyrett
    python scripts/raise_funds_offline_demo.py --buy MU --shares 5

EVERYTHING EXTERNAL IS REPLACED, and the output says so: the quotes are INVENTED round numbers (a fake toolbox behind the real Data Clerk tool and work order), and the "LLM" is a small
scripted function that proposes steps by a fixed rule. The tree search, the hard checks, the scoring, the pruning and the report are the real code. Use it to see the output format and
to try the search; the real run is `python -m prometheus.agents.account_manager.account_manager --buy MU --shares 2`, which calls Finnhub and OpenRouter and costs a little.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage  # noqa: E402

from prometheus.agents.account_manager.account_manager import DEFAULT_PORTFOLIO, AccountManager, report  # noqa: E402
from prometheus.agents.data_clerk.data_clerk_agent import DataClerk  # noqa: E402
from prometheus.common.portfolio import load_portfolio  # noqa: E402

FAKE_PRICES = {"MSFT": 420.00, "TSLA": 250.00, "GM": 55.00, "MU": 410.00, "NVDA": 180.00}  # invented for the demo: they are not market data


class FakeToolbox:
    """Stands in for the DataToolbox: a quote is a fixed number. The real one calls Finnhub's /quote."""

    def get_quote(self, symbol):
        if symbol not in FAKE_PRICES:
            raise RuntimeError(f"No quote returned for {symbol}")
        p = FAKE_PRICES[symbol]
        return {"symbol": symbol, "price": p, "change": 0.0, "change_percent": 0.0, "high": p, "low": p, "open": p, "previous_close": p,
                "as_of_utc": "2026-11-03T21:00:00Z", "source": "FAKE (offline demo)"}


class ScriptedGenerator:
    """The 'LLM' of the demo: for each holding it proposes either the step that alone covers the shortfall or a smaller one that does not, so the tree gets two levels."""
    model_name = "scripted/offline-demo"

    def invoke(self, messages):
        text = messages[-1].content
        short = float(re.search(r"still short \$([\d,\.]+)", text).group(1).replace(",", ""))
        rows = re.findall(r"- (\w+) \(.*?\): held (\d+), sellable (\d+), price \$([\d,\.]+), basis \$([\d,\.]+), (gain|loss) .*?(?:  \(already in the plan\))?$", text, re.M)
        thoughts = []
        for i, (symbol, held, sellable, price, basis, kind) in enumerate(rows):
            if "(already in the plan)" in text.split(f"- {symbol} ")[1].split("\n")[0]:
                continue
            price = float(price.replace(",", ""))
            alone = math.ceil(short / price)
            n = min(alone, int(sellable)) if i % 2 == 0 else max(1, min(alone // 2, int(sellable)))
            thoughts.append({"symbol": symbol, "shares": n, "idea": f"{'cover the shortfall with' if i % 2 == 0 else 'start with a smaller sale of'} {symbol} ({kind} per share)"})
        return AIMessage(content=json.dumps({"thoughts": thoughts}), usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="RaiseFunds with invented quotes and a scripted 'LLM': no network, no key, no cost.")
    parser.add_argument("--buy", default="MU")
    parser.add_argument("--shares", type=int, default=2)
    parser.add_argument("--portfolio", default=str(DEFAULT_PORTFOLIO))
    args = parser.parse_args(argv)
    portfolio = load_portfolio(args.portfolio)
    clerk = DataClerk(toolbox=FakeToolbox(), model=object())  # the real Data Clerk and its real get_quote tool; only the toolbox under it is fake
    agent = AccountManager(model=ScriptedGenerator(), clerk=clerk)
    print("OFFLINE DEMO: invented quotes and a scripted 'LLM'. The search, the checks, the scores and this report are the real code.\n")
    result = agent.raise_funds(portfolio, args.buy, args.shares)
    print(report(result))
    return 0 if result["status"] in ("plan", "funded") else 1


if __name__ == "__main__":
    sys.exit(main())
