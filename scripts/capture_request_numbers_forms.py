"""Capture the REAL replies of the Account Manager's model to requests that say dollars or shares, skip the recommendations, or give a price limit (design choices entry 138; design/request_numbers_spec.md), so that a test can replay them offline.

    python scripts/capture_request_numbers_forms.py                 # says what it would do and stops
    python scripts/capture_request_numbers_forms.py --yes           # does it

WHAT IT CALLS (so it asks for --yes): `AccountManager.understand_request` for each request below, with the Account Manager's own prompt (`prompts/account_manager_request.py`) and its own model (openai/gpt-oss-120b on
OpenRouter, the account_manager role): a few cents in all (one call for a request, two when code refused the first reply). It reads no account, calls no Finnhub, fetches nothing, and writes ONLY
`tests/fixtures/session_real/request_numbers_forms.json`: for each request the raw replies in the order the model gave them (a second reply is its answer to code's refusal of the first), what `parse_intent` made of the
last one (or the reason it was refused), and what the calls cost. Nothing in it is edited, and a request that is already in it is NOT made again (only the new ones are added: entry 140's, and `AGAIN`, under another key). The key is read by `get_chat_model` through `rag.config.setting`; it is never printed or saved.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage  # noqa: E402

from prometheus.agents.account_manager.account_manager import AccountManager  # noqa: E402
from prometheus.common.money import jsonable  # noqa: E402
from prometheus.rag.config import REPO_ROOT  # noqa: E402

OUT = REPO_ROOT / "tests" / "fixtures" / "session_real" / "request_numbers_forms.json"
REQUESTS = (
    "Buy $2500 worth of symbol KO and skip recommendations.",
    "I want to buy $1000 of MU.",
    "I want to buy 10 of MU.",
    "Buy 1000 MU",
    "Buy 1000 dollars of MU",
    "buy 500 bucks of AMD",
    "spend 1000 on MU",
    "Buy the top 3 AI stocks under $50 a share",
    "Buy $1000 of MU under $50 a share",
    "Recommend 3 oil stocks below $40",
    "buy AI stocks, skip recommendations",
    "I want to buy MSFT, no research needed",
    "Buy 10 of MU if the recommendation passes",
    "Buy 10 shares of MU over $20 a share",
    "Buy 5k of MU",
    # the same kinds of request for MSFT, the stock the recorded symbol directory and the stored closes of the end to end tests have (tests/test_request_numbers.py)
    "Buy $2500 worth of symbol MSFT and skip recommendations.",
    "I want to buy 10 of MSFT.",
    "Buy 10 of MSFT under $50 a share",
    "Buy 10 of MSFT under $1000 a share",
    "spend 1000 on MSFT",
    "Buy $5,000 of MSFT and TSLA under $400 a share",
    # entry 140 (design/limit_orders_spec.md): a price floor, "or more" and "or less", and a number of shares typed with a recommendation. Captured with the prompt after entry 140 (the form has `min_price`)
    "Buy the top 3 AI stocks over $20 a share",
    "Recommend 3 oil stocks between $20 and $60 a share",
    "Sell 10 shares of MSFT at $600 or more",
    "Buy 10 of MU at $90 or less",
    "buy MSFT over $20 a share",
    "buy 5 shares of each of the top 3 oil stocks if recommended",
    "Buy 10 of MU and 5 of AMD if they're recommended",
)
# Texts that are already recorded (with the older prompt) and are recorded AGAIN with the prompt after entry 140, under the key "<text> [entry 140]": the old reply stays as it was.
AGAIN = ("Buy 10 of MU if the recommendation passes",)
AGAIN_SUFFIX = " [entry 140]"


class Recording:
    """The real chat model, remembering the raw text of every reply and the tokens it cost."""

    def __init__(self, model):
        self.model, self.model_name = model, getattr(model, "model_name", None)
        self.replies: list[str] = []
        self.tokens_in = self.tokens_out = 0

    def invoke(self, messages):
        reply = self.model.invoke(messages)
        content = reply.content if isinstance(reply.content, str) else ""
        usage = getattr(reply, "usage_metadata", None) or {}
        self.replies.append(content)
        self.tokens_in += usage.get("input_tokens", 0)
        self.tokens_out += usage.get("output_tokens", 0)
        return AIMessage(content=content, usage_metadata=usage)


def capture(typed: str) -> dict:
    manager = AccountManager()
    model = manager.model = Recording(manager.model)
    try:
        intent = manager.understand_request(typed)
        outcome = {"action": intent.action, "describe": intent.describe(), "budget": intent.budget, "each": intent.each, "count": intent.count, "max_price": intent.max_price, "min_price": intent.min_price, "corrected": intent.corrected,
                   "unclear_number": intent.unclear_number, "topic": intent.topic, "symbols": intent.symbols, "shares_of": intent.shares_of, "shares_each": intent.shares_each,
                   "orders": [{"side": o.side, "symbol": o.symbol, "company": o.company, "shares": o.shares, "dollars": o.dollars, "max_price": o.max_price, "min_price": o.min_price} for o in intent.orders]}
    except ValueError as error:
        outcome = {"refused": str(error)}
    return {"replies": model.replies, "outcome": jsonable(outcome), "tokens_in": model.tokens_in, "tokens_out": model.tokens_out}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--yes", action="store_true", help="make the calls (the Account Manager's model on OpenRouter) and write the fixture")
    args = parser.parse_args()
    before = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    done = dict(before.get("requests", {}))      # what is recorded stays exactly as it is: only the requests that are not in the fixture yet are made
    todo = [(typed, typed) for typed in REQUESTS if typed not in done] + [(typed, typed + AGAIN_SUFFIX) for typed in AGAIN if typed + AGAIN_SUFFIX not in done]
    print(f"{len(REQUESTS) + len(AGAIN)} requests, {len(done)} already recorded, {len(todo)} to make -> {OUT.relative_to(REPO_ROOT)}")
    if not args.yes:
        print("(this calls the Account Manager's model once or twice for each request to make: a few cents; run with --yes)")
        return
    for typed, key in todo:
        done[key] = capture(typed)
        print(f"  {key!r}: {len(done[key]['replies'])} repl{'y' if len(done[key]['replies']) == 1 else 'ies'}, {done[key]['outcome'].get('refused') or done[key]['outcome']['describe']}")
    if not todo:
        return
    model_name = AccountManager().model.model_name
    note = before.get("_note") or ("REAL replies of the Account Manager's model (" + str(model_name) + " on OpenRouter, the account_manager role) to requests that say dollars or shares, skip the recommendations or give a price limit, recorded on " + date.today().isoformat()
            + " by AccountManager.understand_request with the Account Manager's own prompt (prompts/account_manager_request.py after design choices entry 138: the form has `max_price`). 'replies' is what the model answered, in order "
            "(a second reply is the answer to code's refusal of the first); 'outcome' what parse_intent made of it. Nothing here is edited. The tests give these replies back to the real Account Manager, whose code checks every field against the user's own words.")
    if "ENTRY 140" not in note:
        note += (f" ENTRY 140 (recorded {date.today().isoformat()}, prompts/account_manager_request.py after design choices entry 140: the form has `min_price` and says that shares typed with a recommendation are 'buy'): the requests from 'Buy the top 3 AI stocks over $20 a share' on, "
                 "and the text recorded again under the key '<text> [entry 140]'; the older requests keep the replies they were recorded with.")
    tokens = (sum(v["tokens_in"] for v in done.values()), sum(v["tokens_out"] for v in done.values()))
    OUT.write_text(json.dumps({"_note": note, "model": model_name, "tokens_in": tokens[0], "tokens_out": tokens[1], "requests": done}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"written: {OUT.relative_to(REPO_ROOT)}; tokens in {tokens[0]:,}, out {tokens[1]:,}")


if __name__ == "__main__":
    main()
