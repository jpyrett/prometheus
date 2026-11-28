"""Capture the REAL replies of the Account Manager's model to news requests that name a PAST DAY or span (design choices entry 151; design/news_dated_window_spec.md, section 3), so that a test can replay them offline.

    python scripts/capture_news_dated_forms.py                 # says what it would do and stops
    python scripts/capture_news_dated_forms.py --yes           # does it

WHAT IT CALLS (so it asks for --yes): `AccountManager.understand_request` for each request below, with the Account Manager's own prompt (`prompts/account_manager_request.py`, after entry 151: the form has `news_from` and `news_to`, and the user message begins
with "Today is Saturday 2026-11-28.") and its own model (openai/gpt-oss-120b on OpenRouter, the account_manager role): a few cents in all (one call for a request, two when code refused the first reply). TODAY IS FIXED (`TODAY`, passed to `understand_request`) so
that "last Tuesday" and "the day before Thanksgiving" have one answer whenever this runs and whenever the fixture is replayed. It reads no account, calls no Finnhub, fetches nothing, and writes ONLY `tests/fixtures/session_real/news_dated_forms.json`: for each request
the raw replies in the order the model gave them (a second reply is its answer to code's refusal of the first), what `parse_intent` made of the last one (or the reason it was refused), and what the calls cost. Nothing in it is edited, and a request that is
already in it is NOT made again. The key is read by `get_chat_model` through `rag.config.setting`; it is never printed or saved. Same shape as `scripts/capture_news_request_forms.py`.
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

OUT = REPO_ROOT / "tests" / "fixtures" / "session_real" / "news_dated_forms.json"
TODAY = date(2026, 11, 28)      # a Saturday: "last Tuesday" is 2026-11-24 and Thanksgiving (the fourth Thursday of November) was 2026-11-26
REQUESTS = (
    # the day in the words: a program reads the date (and wins), the model's own reading is recorded beside it
    "news on TSLA on November 20",
    "What was the news on TSLA on 2026-11-20?",
    "NVDA news from Nov 3 to Nov 5",
    # a day the program does not read: the model works it out from the date in its message
    "what happened with Apple last Tuesday?",
    "TSLA news the day before Thanksgiving",
    # the rolling control: no dates, as entry 149
    "TSLA news last week",
)


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
        intent = manager.understand_request(typed, today=TODAY)
        outcome = {"action": intent.action, "describe": intent.describe(), "topic": intent.topic, "symbols": intent.symbols, "companies": intent.companies, "news_days": intent.news_days, "news_from": intent.news_from, "news_to": intent.news_to,
                   "corrected": intent.corrected}
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
    note = before.get("_note") or ("REAL replies of the Account Manager's model (" + str(model_name) + " on OpenRouter, the account_manager role) to news requests that name a past day or span, and to a rolling control, recorded on " + date.today().isoformat()
            + " by AccountManager.understand_request with the Account Manager's own prompt (prompts/account_manager_request.py after design choices entry 151: the form has `news_from` and `news_to`) and TODAY FIXED at " + TODAY.isoformat()
            + " (the user message begins \"Today is Saturday 2026-11-28.\"). 'replies' is what the model answered, in order (a second reply is the answer to code's refusal of the first); 'outcome' what parse_intent made of it. Nothing here is edited. "
            "The tests give these replies back to the real Account Manager, with the same fixed today, and its code checks every field against the user's own words.")
    tokens = (sum(v["tokens_in"] for v in done.values()), sum(v["tokens_out"] for v in done.values()))
    OUT.write_text(json.dumps({"_note": note, "model": model_name, "today": TODAY.isoformat(), "tokens_in": tokens[0], "tokens_out": tokens[1], "requests": done}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"written: {OUT.relative_to(REPO_ROOT)}; tokens in {tokens[0]:,}, out {tokens[1]:,}")


if __name__ == "__main__":
    main()
