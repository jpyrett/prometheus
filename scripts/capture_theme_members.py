"""Capture the REAL replies of the Data Clerk's model to "does this company belong to the theme?" (design choices entry 120; design/theme_membership_spec.md), so that a test can replay them offline.

    python scripts/capture_theme_members.py                 # says what it would do and stops
    python scripts/capture_theme_members.py --yes           # does it

WHAT IT CALLS (so it asks for --yes): for each of the two phrases of the real sessions of 2026-11-20 ("oil" and "cybersecurity") it runs `DataClerk.find_companies` for real, as the desk does:
  - the Data Clerk's model on OpenRouter: the proposal call and the membership call (a few cents in all; a reply that is not valid JSON is tried once more by `calls.ask_object`);
  - Finnhub's symbol directory (read only, the free search endpoint, paced): the names of the proposed companies and of the tickers the stored news names;
  - the STORED market news (`discover_tickers`, free): it reads `news/market/` as it stands today and writes nothing.
It fetches NOTHING and writes ONLY `tests/fixtures/data_clerk_real/theme_members.json`: for each phrase the discovery's rows, every directory query with its rows, the raw replies of the model in the order they were asked,
what `find_companies` returned, and the model's name and tokens. Nothing in it is edited. The keys are read by `get_chat_model` and `search_finnhub` through `rag.config.setting`; they are never printed or saved.
"""
import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage  # noqa: E402

from prometheus.agents.data_clerk.data_clerk_agent import DataClerk  # noqa: E402
from prometheus.models.chat import get_chat_model  # noqa: E402
from prometheus.rag.config import REPO_ROOT  # noqa: E402
from prometheus.tools import symbols  # noqa: E402
from prometheus.tools.data_clerk import DataClerkTools  # noqa: E402

PHRASES = ("oil", "cybersecurity")      # "I want to buy oil stocks..." and "the 3 most recommended cybersecurity stocks..." (2026-11-20)
COUNT = 10                              # what the desk asks for a request of 3 (`recommendations.find_companies_of`: max(10, find * 2))
OUT = REPO_ROOT / "tests" / "fixtures" / "data_clerk_real" / "theme_members.json"


class Recording:
    """The real chat model, remembering the raw text of every reply and the tokens it cost."""

    def __init__(self, model):
        self.model, self.model_name = model, getattr(model, "model_name", None)
        self.calls: list[dict] = []

    def invoke(self, messages):
        reply = self.model.invoke(messages)
        content = reply.content if isinstance(reply.content, str) else ""
        usage = getattr(reply, "usage_metadata", None) or {}
        stage = "membership" if "BELONGS to the theme" in messages[0].content else "proposal"
        self.calls.append({"stage": stage, "reply": content, "tokens_in": usage.get("input_tokens", 0), "tokens_out": usage.get("output_tokens", 0)})
        return AIMessage(content=content, usage_metadata=usage)


class RecordingSearch:
    """Finnhub's symbol search, remembering every answer by its query."""

    def __init__(self):
        self.rows: dict[str, list] = {}

    def __call__(self, query: str):
        rows = symbols.search_finnhub(query)
        self.rows[query] = rows
        return rows


class RecordingTools:
    """The Data Clerk's tools, remembering what the stored-news discovery returned."""

    def __init__(self, tools):
        self.tools, self.discovered = tools, []

    def discover_tickers(self, phrase, from_date=7, top=10):
        found = self.tools.discover_tickers(phrase, from_date=from_date, top=top)
        self.discovered.append({"phrase": phrase, "from_date": from_date, "top": top, "scanned": found.get("scanned"), "matched": found.get("matched"), "tickers": found["tickers"]})
        return found


def capture(phrase: str, today: date) -> dict:
    model, search, tools = Recording(get_chat_model("data_clerk")), RecordingSearch(), RecordingTools(DataClerkTools())
    clerk = DataClerk(model=model, symbol_search=search)
    clerk.tools = tools
    result = clerk.find_companies(phrase, count=COUNT, today=today)
    return {"phrase": phrase, "model": model.model_name, "count": COUNT, "today": today.isoformat(), "discovered": tools.discovered[0], "directory": search.rows, "model_calls": model.calls, "result": result}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true", help="really call the model (OpenRouter, paid, a few cents) and Finnhub's symbol directory")
    args = parser.parse_args(argv)
    print(f"{len(PHRASES)} phrases ({', '.join(PHRASES)}) -> DataClerk.find_companies with the real model, the real symbol directory and the stored market news")
    if not args.yes:
        print("nothing was called: add --yes to call the model (OpenRouter, paid) and Finnhub. It writes only " + str(OUT.relative_to(REPO_ROOT)))
        return 0
    today = datetime.now(timezone.utc).date()
    cases = {}
    try:
        for phrase in PHRASES:
            cases[phrase] = capture(phrase, today)
            c = cases[phrase]
            asked = [n for n in c["model_calls"] if n["stage"] == "membership"]
            print(f"  {phrase}: {len(c['model_calls'])} model calls ({len(asked)} membership), {len(c['result']['companies'])} companies, {len(c['result']['notes'])} notes")
    except RuntimeError as error:      # a missing key: the message names the setting, never a value
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    calls = [n for c in cases.values() for n in c["model_calls"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "_note": "REAL replies of the Data Clerk's model (the data_clerk role, temperature 0) to the two questions of DataClerk.find_companies for 'oil' and 'cybersecurity', captured with scripts/capture_theme_members.py (design choices entry 120): "
                 "the PROPOSAL (which companies belong to the phrase) and the MEMBERSHIP (does each ticker the stored market news names for the phrase belong to it). `discovered` is what the stored market news gave (discover_tickers, 30 days, top 10), "
                 "`directory` every Finnhub symbol search that was made with its rows, `model_calls` the raw replies in the order they were asked, `result` what find_companies returned. Nothing is edited.",
        "model": next(iter(cases.values()))["model"], "captured_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tokens_in": sum(n["tokens_in"] for n in calls), "tokens_out": sum(n["tokens_out"] for n in calls), "cases": cases}, indent=1, ensure_ascii=True), encoding="utf-8")
    print(f"saved {OUT.relative_to(REPO_ROOT)}: {len(calls)} model calls, {sum(n['tokens_in'] for n in calls)} tokens in, {sum(n['tokens_out'] for n in calls)} tokens out")
    return 0


if __name__ == "__main__":
    sys.exit(main())
