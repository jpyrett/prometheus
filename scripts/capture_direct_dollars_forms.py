"""Capture the REAL replies of the Account Manager's model to requests that give an amount of money (design choices entry 122; design/direct_dollars_each_spec.md), so that a test can replay them offline.

    python scripts/capture_direct_dollars_forms.py                 # says what it would do and stops
    python scripts/capture_direct_dollars_forms.py --yes           # does it
    python scripts/capture_direct_dollars_forms.py --each --yes    # the second capture (entry 125, design/dollars_each_word_spec.md): the requests with the word "each", and every request of the first capture again

WHAT IT CALLS (so it asks for --yes): `AccountManager.understand_request` for each request below, with the Account Manager's own prompt (`prompts/account_manager_request.py`) and its own model (openai/gpt-oss-120b on
OpenRouter, the account_manager role): a few cents in all (one call for a request, two when code refused the first reply). It reads no account, calls no Finnhub, fetches nothing, and writes ONLY
`tests/fixtures/session_real/direct_dollars_forms.json`: for each request the raw replies in the order the model gave them (a second reply is its answer to code's refusal of the first), what `parse_intent` made of the
last one (or the reason it was refused), and what the calls cost. Nothing in it is edited. The key is read by `get_chat_model` through `rag.config.setting`; it is never printed or saved.

The requests are the six of the spec, every request of the earlier captures (`direct_order_forms.json`, `direct_order_decimal_forms.json`: the prompt changed, so each reading is made again) and the four theme requests of
`theme_request_forms.json` (the budget line of the prompt changed too).

WITH `--each` (design choices entry 125) the prompt has the word "each" (`TradeIntent.each`), so every request is asked again with it: the eight requests of `design/dollars_each_word_spec.md` (the five that say each and the three whose
reading must not change) and every request of the first capture, into `direct_dollars_each_word_forms.json`. The first capture file is never edited.
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

SESSION_REAL = REPO_ROOT / "tests" / "fixtures" / "session_real"
OUT = SESSION_REAL / "direct_dollars_forms.json"
OUT_EACH = SESSION_REAL / "direct_dollars_each_word_forms.json"
EACH_NEW = ("Buy $3,000 each of XOM and MSFT", "Buy $3,000 of XOM and MSFT each", "Put $2,000 apiece into NVDA and AMD", "Buy 10 shares each of AAPL and MSFT", "Buy the top 3 oil stocks with $3,000 each",
            "Buy $5,000 of MU and AMD", "Buy $3,000 of XOM and $3,000 of MSFT", "Buy the 3 most recommended EV stocks with $5,000")      # the last three must not change their reading
NEW = ("Buy $3,000 of XOM and $3,000 of MSFT", "Buy $5,000 of MSFT", "Buy $2,000 of AAPL and 10 shares of MSFT", "Put $1,500 into NVDA and $500 into AMD", "Buy $5,000 of MU and AMD", "Buy $2,000 of AAPL and MSFT")
EARLIER = ("direct_order_forms.json", "direct_order_decimal_forms.json", "theme_request_forms.json")


def requests(each: bool = False) -> list[str]:
    """The six new requests, then every request of the earlier captures, each once, in the order they are recorded. With `each` (entry 125): the requests of the word "each" first, then every request of the first capture."""
    texts = list(EACH_NEW) if each else []
    for typed in NEW:
        if typed not in texts:
            texts.append(typed)
    for name in EARLIER:
        for typed in json.loads((SESSION_REAL / name).read_text(encoding="utf-8"))["requests"]:
            if typed not in texts:
                texts.append(typed)
    return texts


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
        outcome = {"action": intent.action, "describe": intent.describe(), "budget": intent.budget, "each": intent.each, "count": intent.count, "orders": [{"side": o.side, "symbol": o.symbol, "company": o.company, "shares": o.shares, "dollars": o.dollars} for o in intent.orders]}
    except ValueError as error:
        outcome = {"refused": str(error)}
    return {"replies": model.replies, "outcome": jsonable(outcome), "tokens_in": model.tokens_in, "tokens_out": model.tokens_out}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--yes", action="store_true", help="make the calls (the Account Manager's model on OpenRouter) and write the fixture")
    parser.add_argument("--each", action="store_true", help="the second capture of entry 125 (the word each): writes direct_dollars_each_word_forms.json")
    args = parser.parse_args()
    texts, out = requests(args.each), OUT_EACH if args.each else OUT
    print(f"{len(texts)} requests -> {out.relative_to(REPO_ROOT)}")
    if not args.yes:
        print("(this calls the Account Manager's model once or twice for each request: a few cents; run with --yes)")
        return
    done, model_name = {}, None
    for typed in texts:
        done[typed] = capture(typed)
        print(f"  {typed!r}: {len(done[typed]['replies'])} repl{'y' if len(done[typed]['replies']) == 1 else 'ies'}, {done[typed]['outcome'].get('refused') or done[typed]['outcome']['describe']}")
    model_name = AccountManager().model.model_name
    if args.each:
        note = ("REAL replies of the Account Manager's model (" + str(model_name) + " on OpenRouter, the account_manager role) to requests that give an amount of money, recorded on " + date.today().isoformat()
                + " by AccountManager.understand_request with the Account Manager's own prompt (prompts/account_manager_request.py after design choices entry 125: the form has `each`): the eight requests of design/dollars_each_word_spec.md "
                "(the five that say each, and the three whose reading must not change), then every request of direct_dollars_forms.json again (the prompt changed). 'replies' is what the model answered, in order (a second reply is the answer to "
                "code's refusal of the first); 'outcome' what parse_intent made of it. Nothing here is edited. The tests give these replies back to the real Account Manager, whose code checks every field against the user's own words.")
    else:
        note = ("REAL replies of the Account Manager's model (" + str(model_name) + " on OpenRouter, the account_manager role) to requests that give an amount of money, recorded on " + date.today().isoformat()
            + " by AccountManager.understand_request with the Account Manager's own prompt (prompts/account_manager_request.py after design choices entry 122: an order has `dollars`): the six requests of design/direct_dollars_each_spec.md, "
            "every request of direct_order_forms.json and direct_order_decimal_forms.json (the prompt changed, so each reading is made again) and the four theme requests of theme_request_forms.json. 'replies' is what the model answered, in order "
            "(a second reply is the answer to code's refusal of the first); 'outcome' what parse_intent made of it. Nothing here is edited. The tests give these replies back to the real Account Manager, whose code checks every field against the user's own words.")
    tokens = (sum(v["tokens_in"] for v in done.values()), sum(v["tokens_out"] for v in done.values()))
    out.write_text(json.dumps({"_note": note, "model": model_name, "tokens_in": tokens[0], "tokens_out": tokens[1], "requests": done}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"written: {out.relative_to(REPO_ROOT)}; tokens in {tokens[0]:,}, out {tokens[1]:,}")


if __name__ == "__main__":
    main()
