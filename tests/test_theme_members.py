"""A company found in the stored market news joins a theme only when the Data Clerk's model confirms that it belongs (design choices entry 120; design/theme_membership_spec.md).

THE PROBLEM, from two real sessions of 2026-11-20: "oil stocks" bought Procter & Gamble (a headline about oil prices names it) and "cybersecurity" bought Amazon. `DataClerk.find_companies` added every ticker that the stored
market news ties to the phrase with 3 or more articles, with no check that the company belongs to the theme.

REAL DATA ONLY: `tests/fixtures/data_clerk_real/theme_members.json` was made by `scripts/capture_theme_members.py` on 2026-11-20: `DataClerk.find_companies("oil")` and `("cybersecurity")` run for real, with the stored market news
(`discover_tickers`) as it stood that day, Finnhub's symbol directory and the Data Clerk's real model (openai/gpt-oss-120b, temperature 0); the raw replies of the model, the directory's rows and the discovery's rows are in it,
nothing edited. These tests give them back to the real `DataClerk`, whose code decides what is added. A test of the code's own checks edits ONE field of the real membership reply.
"""
import json
import re
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from prometheus.agents.data_clerk.data_clerk_agent import NEWS_ADDED, DataClerk
from prometheus.tools import symbols

CAPTURE = json.loads((Path(__file__).parent / "fixtures" / "data_clerk_real" / "theme_members.json").read_text(encoding="utf-8"))
CASES = CAPTURE["cases"]
OIL, CYBER = CASES["oil"], CASES["cybersecurity"]


class Replay:
    """The Data Clerk's model: its REAL recorded replies, given back in the order they were asked (proposal first, then membership)."""

    model_name = CAPTURE["model"]

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def invoke(self, messages):
        self.seen.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(content=reply, usage_metadata={"input_tokens": 10, "output_tokens": 10})


class Directory:
    """Finnhub's symbol search as recorded on 2026-11-20. A query that was never made then is a test that left the recorded path, and fails. `broken` are queries the lookup fails for."""

    def __init__(self, case, broken=(), unrecorded_finds_nothing=False):
        self.rows, self.broken, self.asked, self.lenient = case["directory"], set(broken), [], unrecorded_finds_nothing

    def __call__(self, query):
        self.asked.append(query)
        if query in self.broken:
            raise symbols.SymbolLookupError("the symbol lookup failed (HTTP 500)", reason="the symbol lookup failed (HTTP 500)")
        return self.rows.get(query, []) if self.lenient else self.rows[query]


class Tools:
    """What the stored market news gave on 2026-11-20 (`discover_tickers`, 30 days, top 10), as recorded."""

    def __init__(self, case, only=None):
        rows = case["discovered"]["tickers"]
        self.rows = [r for r in rows if only is None or r["symbol"] in only]

    def discover_tickers(self, phrase, from_date=7, top=10):
        return {"tickers": self.rows}


def replies_of(case, edit=None, membership=None):
    """The recorded raw replies of a case, in order. `edit(members)` changes the membership reply's list; `membership` replaces its text (ONE reply, edited by one field)."""
    texts = [c["reply"] for c in case["model_calls"]]
    if membership is not None:
        texts[1] = membership
    elif edit is not None:
        data = json.loads(texts[1])
        edit(data["members"])
        texts[1] = json.dumps(data)
    return texts


def run(case, replies=None, directory=None, tools=None):
    model = Replay(replies if replies is not None else replies_of(case))
    clerk = DataClerk(model=model, symbol_search=directory or Directory(case))
    clerk.lookup_interval = 0
    clerk.tools = tools or Tools(case)
    out = clerk.find_companies(case["phrase"], count=case["count"], today=date.fromisoformat(case["today"]))
    return out, model


def symbols_of(out):
    return [c["symbol"] for c in out["companies"]]


def news_rows(out):
    return {c["symbol"]: c for c in out["companies"] if c["how"] == NEWS_ADDED}


# ---------------------------------------------------------------- the real replies


def test_a_ticker_the_symbol_lookup_fails_for_is_not_sent_and_the_failure_is_said():
    out, model = run(OIL, directory=Directory(OIL, broken={"PG"}))
    sent = json.loads(re.search(r"<tickers>\n(.*)\n</tickers>", model.seen[1][1].content, re.S).group(1))
    assert "PG" not in [s["symbol"] for s in sent] and "PG" not in symbols_of(out)
    assert any(n.startswith("PG: ") and "the symbol lookup failed" in n for n in out["notes"])


# ---------------------------------------------------------------- code checks the form of the answer (T3), on the real reply edited by one field


def change(of, **fields):
    return lambda members: next(m for m in members if m["symbol"] == of).update(fields)


@pytest.mark.parametrize("belongs", ["true", "yes", 1, None])
def test_a_belongs_that_is_not_true_or_false_is_not_a_yes(belongs):
    out, _ = run(OIL, replies_of(OIL, edit=change("CVE", belongs=belongs)))
    assert "CVE" not in symbols_of(out) and any(n.startswith("CVE (") and "did not say whether it belongs" in n for n in out["notes"])


def test_a_reply_that_is_not_the_shape_is_tried_once_more_then_the_news_adds_nothing_and_it_says_so():
    good = OIL["model_calls"][1]["reply"]
    retried, _ = run(OIL, replies=[OIL["model_calls"][0]["reply"], "not json at all", good])
    assert retried["companies"] == OIL["result"]["companies"] and retried["membership"]["calls"] == 2 and retried["membership"]["error"] is None
    failed, model = run(OIL, replies=[OIL["model_calls"][0]["reply"], "not json at all", json.dumps({"members": "no list"})])
    assert news_rows(failed) == {} and symbols_of(failed) == [c["symbol"] for c in OIL["result"]["companies"] if c["how"] != NEWS_ADDED]
    assert failed["membership"]["calls"] == 2 and "not usable" in failed["membership"]["error"] and len(model.seen) == 3
    assert any("none was added" in n and "could not confirm that they belong" in n and "PG" in n for n in failed["notes"])
