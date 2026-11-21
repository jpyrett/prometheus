"""Capture the REAL replies of the Account Manager's model to requests that say "raise funds if needed" (design choices entry 142; design/direct_raise_funds_spec.md, R2, R10 and R12), so that a test can replay them offline.

    python scripts/capture_raise_funds_forms.py                 # says what it would do and stops
    python scripts/capture_raise_funds_forms.py --yes           # does it

A sibling of `scripts/capture_request_numbers_forms.py` (the same recording model and the same shape of file). WHAT IT CALLS (so it asks for --yes): `AccountManager.understand_request` for each request below, with the Account
Manager's own prompt (`prompts/account_manager_request.py`, after entry 142: the form has `raise_funds`) and its own model (openai/gpt-oss-120b on OpenRouter, the account_manager role): a few cents in all (one call for a request, two when
code refused the first reply). It reads no account, calls no Finnhub, fetches nothing, and writes ONLY `tests/fixtures/session_real/raise_funds_forms.json`: for each request the raw replies in the order the model gave them (a second
reply is its answer to code's refusal of the first), what `parse_intent` made of the last one (or the reason it was refused), and what the calls cost. Nothing in it is edited, and a request that is already in it is NOT made again.
The key is read by `get_chat_model` through `rag.config.setting`; it is never printed or saved.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from capture_request_numbers_forms import Recording  # noqa: E402

from prometheus.agents.account_manager.account_manager import AccountManager  # noqa: E402
from prometheus.common.money import jsonable  # noqa: E402
from prometheus.rag.config import REPO_ROOT  # noqa: E402

OUT = REPO_ROOT / "tests" / "fixtures" / "session_real" / "raise_funds_forms.json"
REQUESTS = (
    # a direct order (R2): the words that give permission, the words that do not, and the words that take it back
    "Buy 2 shares of ANT and raise funds if needed",
    "Buy 2 shares of ANT",
    "Buy 2 shares of ANT but don't sell anything",
    "Buy 2 shares of ANT, don't raise funds",
    "buy 5 MSFT, sell some stock to pay for it if you have to",
    "Buy 2 shares of ANT and 3 shares of MSFT and raise funds if needed",
    "Buy $500 of MSFT and raise funds if needed",
    "Sell 10 shares of MSFT",
    # the same kinds for ANT, the stock of Jacob's account 6553 whose quote is recorded with its holdings (tests/fixtures/raise_funds/account_6553.json)
    "Buy $200 of ANT and raise funds if needed",
    "Buy 2 shares of ANT and 1 share of GOOGL and raise funds if needed",
    # shares typed with a recommendation (R10): the permission goes with them
    "buy 10 of MU if recommended and raise funds if needed",
    "Buy 5 shares of each of the top 3 oil stocks if recommended, and raise funds if needed",
    "buy 10 of MU if the recommendation passes, but don't sell anything",
    # entry 143: a sale and a buy in the same request (S1 to S3), and a typed purchase the sales may fall short of paying for (B1)
    "Sell 1 share of MCD and buy 3 shares of ANT",
    "Sell 1 share of MCD and buy 5 shares of ANT and raise funds if needed",
    "Buy 4 shares of ANT and raise funds if needed",
    "Sell 1 share of MCD and buy 5 shares of ANT",
    "Sell 5 shares of MCD and buy 3 shares of ANT",
    # entry 143, W-F: a purchase that WAITS for its limit, with and without the words that permit raising funds when the price is reached
    "Buy 2 shares of ANT at $85 or less and raise funds if needed",
    "Buy 2 shares of ANT at $85 or less",
)


def capture(typed: str) -> dict:
    manager = AccountManager()
    model = manager.model = Recording(manager.model)
    try:
        intent = manager.understand_request(typed)
        outcome = {"action": intent.action, "describe": intent.describe(), "raise_funds": intent.raise_funds, "budget": intent.budget, "each": intent.each, "count": intent.count, "corrected": intent.corrected, "topic": intent.topic,
                   "symbols": intent.symbols, "shares_of": intent.shares_of, "shares_each": intent.shares_each,
                   "orders": [{"side": o.side, "symbol": o.symbol, "company": o.company, "shares": o.shares, "dollars": o.dollars} for o in intent.orders]}
    except ValueError as error:
        outcome = {"refused": str(error)}
    return {"replies": model.replies, "outcome": jsonable(outcome), "tokens_in": model.tokens_in, "tokens_out": model.tokens_out}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--yes", action="store_true", help="make the calls (the Account Manager's model on OpenRouter) and write the fixture")
    args = parser.parse_args()
    before = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    done = dict(before.get("requests", {}))      # what is recorded stays exactly as it is: only the requests that are not in the fixture yet are made
    todo = [typed for typed in REQUESTS if typed not in done]
    print(f"{len(REQUESTS)} requests, {len(done)} already recorded, {len(todo)} to make -> {OUT.relative_to(REPO_ROOT)}")
    if not args.yes:
        print("(this calls the Account Manager's model once or twice for each request to make: a few cents; run with --yes)")
        return
    for typed in todo:
        done[typed] = capture(typed)
        print(f"  {typed!r}: {len(done[typed]['replies'])} repl{'y' if len(done[typed]['replies']) == 1 else 'ies'}, {done[typed]['outcome'].get('refused') or done[typed]['outcome']['describe']}")
    if not todo:
        return
    model_name = AccountManager().model.model_name
    note = before.get("_note") or ("REAL replies of the Account Manager's model (" + str(model_name) + " on OpenRouter, the account_manager role) to requests that say 'raise funds if needed', recorded on " + date.today().isoformat()
            + " by AccountManager.understand_request with the Account Manager's own prompt (prompts/account_manager_request.py after design choices entry 142: the form has `raise_funds`). 'replies' is what the model answered, in order "
            "(a second reply is the answer to code's refusal of the first); 'outcome' what parse_intent made of it. Nothing here is edited. The tests give these replies back to the real Account Manager, whose code checks every field against the user's own words.")
    tokens = (sum(v["tokens_in"] for v in done.values()), sum(v["tokens_out"] for v in done.values()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"_note": note, "model": model_name, "tokens_in": tokens[0], "tokens_out": tokens[1], "requests": done}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"written: {OUT.relative_to(REPO_ROOT)}; tokens in {tokens[0]:,}, out {tokens[1]:,}")


if __name__ == "__main__":
    main()
