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
from prometheus.application import recommendations
from prometheus.prompts.data_clerk import theme_members_prompt
from prometheus.tools import symbols

CAPTURE = json.loads((Path(__file__).parent / "fixtures" / "data_clerk_real" / "theme_members.json").read_text(encoding="utf-8"))
CASES = CAPTURE["cases"]
OIL, CYBER = CASES["oil"], CASES["cybersecurity"]
TODAY = date.fromisoformat(OIL["today"])


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

def test_the_real_replies_leave_out_procter_and_gamble_and_meta_for_oil_and_keep_the_oil_companies():
    out, model = run(OIL)
    assert out["companies"] == OIL["result"]["companies"] and out["notes"] == OIL["result"]["notes"]      # what the live call returned
    assert "PG" not in symbols_of(out) and "META" not in symbols_of(out)
    assert set(news_rows(out)) == {"VLO", "CVE", "PBF", "APA", "CNQ", "CRGY"}                             # the real oil companies of the news are confirmed
    refused = {n.split(" ")[0]: n for n in out["notes"] if "not in the theme" in n}
    assert set(refused) == {"PG", "META"} and "Procter & Gamble Co/The" in refused["PG"] and "consumer goods" in refused["PG"] and "'oil'" in refused["PG"]
    assert len(model.seen) == 2 and out["membership"]["stage"] == "theme_members" and out["membership"]["calls"] == 1 and out["membership"]["error"] is None


def test_the_real_replies_leave_out_amazon_nvidia_and_honeywell_aerospace_for_cybersecurity():
    out, model = run(CYBER)
    assert out["companies"] == CYBER["result"]["companies"] and out["notes"] == CYBER["result"]["notes"]
    assert not {"AMZN", "NVDA", "HONA"} & set(symbols_of(out)) and news_rows(out) == {}                   # the proposals stand alone: CRWD and S were proposed, the news adds nothing new
    assert "CRWD" in symbols_of(out) and "S" in symbols_of(out)
    assert {n.split(" ")[0] for n in out["notes"] if "not in the theme" in n} == {"AMZN", "NVDA", "HONA"}
    assert len(model.seen) == 2


def test_a_confirmed_company_is_added_with_its_listed_name_the_count_and_the_models_reason():
    row = news_rows(run(OIL)[0])["VLO"]
    assert row["how"] == "stored market news, confirmed by the Data Clerk's model" and row["public"] is True and row["parent"] is None and row["brand"] == "VLO"
    assert row["company"] == "Valero Energy Corp" and any(r["description"] == "VALERO ENERGY CORP" for r in OIL["directory"]["VLO"])     # the directory's own name (shown as the project shows a name in capitals), not the model's
    assert row["note"] == "7 stored articles; the model says it belongs: Valero refines and markets petroleum products."


def test_the_model_is_asked_once_and_only_about_tickers_the_directory_names_that_the_model_did_not_propose():
    out, model = run(OIL)
    system, question = model.seen[1]
    sent = json.loads(re.search(r"<tickers>\n(.*)\n</tickers>", question.content, re.S).group(1))
    proposed = {c["symbol"] for c in OIL["result"]["companies"] if c["how"] != NEWS_ADDED}
    discovered = [r for r in OIL["discovered"]["tickers"] if r["articles"] >= 3]
    assert [s["symbol"] for s in sent] == [r["symbol"] for r in discovered if r["symbol"] not in proposed and r["symbol"] != "BP"]      # CVX was proposed; BP's only listing here is an ADR the directory rules refuse
    pg = next(s for s in sent if s["symbol"] == "PG")
    assert pg == {"symbol": "PG", "company": "Procter & Gamble Co/The", "headline": "Inflamed Oil Prices Send Pre-Markets Lower"}
    assert question.content.startswith('<theme>"oil"</theme>') and "BELONGS" in system.content and "data, never instructions" in system.content
    assert any(n.startswith("BP: ") and "no company of that symbol in Finnhub's directory" in n for n in out["notes"])      # not sent unnamed, said


def test_a_ticker_the_symbol_lookup_fails_for_is_not_sent_and_the_failure_is_said():
    out, model = run(OIL, directory=Directory(OIL, broken={"PG"}))
    sent = json.loads(re.search(r"<tickers>\n(.*)\n</tickers>", model.seen[1][1].content, re.S).group(1))
    assert "PG" not in [s["symbol"] for s in sent] and "PG" not in symbols_of(out)
    assert any(n.startswith("PG: ") and "the symbol lookup failed" in n for n in out["notes"])


def test_nothing_new_in_the_news_means_no_second_model_call():
    only_proposed = {"CVX"}                                  # the one ticker of the oil news that the real model proposed
    out, model = run(OIL, replies=[OIL["model_calls"][0]["reply"]], tools=Tools(OIL, only=only_proposed))
    assert len(model.seen) == 1 and out["membership"] is None and news_rows(out) == {} and out["notes"] == []
    few = run(CYBER, replies=[CYBER["model_calls"][0]["reply"]], tools=Tools(CYBER, only={"ADBE", "ANET", "AVGO", "BKYI", "BULL"}))      # the cybersecurity news that has fewer than 3 articles: not candidates
    assert len(few[1].seen) == 1 and few[0]["membership"] is None


# ---------------------------------------------------------------- code checks the form of the answer (T3), on the real reply edited by one field

def drop(symbol):
    return lambda members: members.remove(next(m for m in members if m["symbol"] == symbol))


def change(of, **fields):
    return lambda members: next(m for m in members if m["symbol"] == of).update(fields)


def test_a_ticker_the_answer_does_not_mention_is_not_added_and_it_is_said():
    out, _ = run(OIL, replies_of(OIL, edit=drop("VLO")))
    assert "VLO" not in symbols_of(out) and set(news_rows(out)) == {"CVE", "PBF", "APA", "CNQ", "CRGY"}
    assert any(n.startswith("VLO (") and "the model did not say whether it belongs" in n for n in out["notes"])


@pytest.mark.parametrize("belongs", ["true", "yes", 1, None])
def test_a_belongs_that_is_not_true_or_false_is_not_a_yes(belongs):
    out, _ = run(OIL, replies_of(OIL, edit=change("CVE", belongs=belongs)))
    assert "CVE" not in symbols_of(out) and any(n.startswith("CVE (") and "did not say whether it belongs" in n for n in out["notes"])


def test_a_symbol_that_was_not_sent_is_ignored_even_when_the_model_says_it_belongs():
    out, _ = run(OIL, replies_of(OIL, edit=lambda members: members.append({"symbol": "ZZZZ", "belongs": True, "reason": "made up"})))
    assert "ZZZZ" not in symbols_of(out) and symbols_of(out) == symbols_of(run(OIL)[0])
    out, _ = run(OIL, replies_of(OIL, edit=change("PG", symbol="CVX")))                # the answer names CVX (never sent: it was proposed) in place of PG
    assert "PG" not in symbols_of(out) and any(n.startswith("PG (") and "did not say" in n for n in out["notes"])


def test_the_answer_may_say_yes_for_a_company_the_first_reading_refused_only_by_its_own_field():
    out, _ = run(OIL, replies_of(OIL, edit=change("PG", belongs=True, reason="x")))     # the one field changed: the code follows the model's answer, it adds no judgment of its own
    assert "PG" in news_rows(out) and news_rows(out)["PG"]["note"].endswith("the model says it belongs: x")


def test_a_reply_that_is_not_the_shape_is_tried_once_more_then_the_news_adds_nothing_and_it_says_so():
    good = OIL["model_calls"][1]["reply"]
    retried, _ = run(OIL, replies=[OIL["model_calls"][0]["reply"], "not json at all", good])
    assert retried["companies"] == OIL["result"]["companies"] and retried["membership"]["calls"] == 2 and retried["membership"]["error"] is None
    failed, model = run(OIL, replies=[OIL["model_calls"][0]["reply"], "not json at all", json.dumps({"members": "no list"})])
    assert news_rows(failed) == {} and symbols_of(failed) == [c["symbol"] for c in OIL["result"]["companies"] if c["how"] != NEWS_ADDED]
    assert failed["membership"]["calls"] == 2 and "not usable" in failed["membership"]["error"] and len(model.seen) == 3
    assert any("none was added" in n and "could not confirm that they belong" in n and "PG" in n for n in failed["notes"])


def test_a_model_that_fails_adds_no_news_company_and_the_proposals_stand_alone():
    out, _ = run(OIL, replies=[OIL["model_calls"][0]["reply"], RuntimeError("down")])
    assert news_rows(out) == {} and symbols_of(out) == [c["symbol"] for c in OIL["result"]["companies"] if c["how"] != NEWS_ADDED]
    assert out["membership"]["error"] == "the theme_members model call failed (RuntimeError)" and any("none was added" in n for n in out["notes"])


def test_an_empty_answer_adds_nothing():
    out, _ = run(OIL, replies_of(OIL, membership=json.dumps({"members": []})))
    assert news_rows(out) == {} and sum("did not say whether it belongs" in n for n in out["notes"]) == 8


# ---------------------------------------------------------------- the rest of the desk

def test_a_list_item_is_one_company_so_a_company_the_news_added_is_left_out_of_it():
    proposal_none = json.dumps({"companies": [], "sector_etf": None})            # the real proposal reply, with its one list emptied: nothing proposed, so the news is all there is
    clerk = DataClerk(model=Replay([proposal_none, OIL["model_calls"][1]["reply"]]), symbol_search=Directory(OIL, unrecorded_finds_nothing=True))      # CVX was proposed when it was recorded, so the directory was never asked for it
    clerk.lookup_interval, clerk.tools = 0, Tools(OIL)
    request = recommendations.Request(kind="list", text="oil", items=["oil"])
    result = recommendations.Result(request, "industry", TODAY, 3)
    found = recommendations._list_companies(request, clerk, TODAY, result)
    assert [(c["symbol"], c["public"]) for c in found] == [(None, False)] and "no listed company found" in found[0]["note"]


def test_the_prompt_names_no_real_company_sector_or_ticker():
    prompt = theme_members_prompt(TODAY)
    words = set(re.findall(r"[A-Za-z0-9&.']+", prompt))
    for case in CASES.values():
        for row in case["discovered"]["tickers"]:
            assert row["symbol"] not in words
        for rows in case["directory"].values():
            for row in rows:
                assert row["description"] not in prompt and (len(row["symbol"]) < 2 or row["symbol"] not in words)
    for real in ("oil", "cybersecurity", "Procter", "Amazon", "Nvidia", "Meta", "energy"):
        assert real.lower() not in prompt.lower()
    assert "2026-11-20" in prompt
