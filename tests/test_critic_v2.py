"""Critic v2 (design/critic_v2_evidence_spec.md): a denial reason must earn its place. The rules C2 to C6, on the REAL captures of the Critic's tests (`tests/fixtures/advisor_real`: PG, PG_open), the REAL stored articles of `news/`, the
real names of the symbol directory and the REAL research of the Time Machine (`tests/fixtures/proving_real/*_cache`: the records and reviews of four oil decision weeks, made by Critic version 1).

C1 is the report card (`tests/test_proving_report_card.py`); C3 is amended in `test_critic_evidence.py` and `test_critic_real.py`; C7 (no size, no portfolio, no liquidity rule, nothing of the Advisor's scoring code) is the Critic's existing tests.
"""
import glob
import json
from pathlib import Path

from advisor_helpers import MemoryLedger
from critic_helpers import ADVISOR_CONFIG, CONFIG, NOW, FakeClerk, ReviewerModel, captured, rec_of, review, tampered_order, view
from test_critic_evidence import Setup

from prometheus.agents.critic import names, recheck as rc
from prometheus.agents.critic import evidence as ev
from prometheus.agents.critic.critic import Critic
from prometheus.agents.critic.labels import LabelStore, open_store, prompt_version
from prometheus.rag.config import REPO_ROOT
from prometheus.rag.loaders import load_news

REAL_CACHES = sorted(glob.glob(str(REPO_ROOT / "tests" / "fixtures" / "proving_real" / "*_cache" / "*.json")))


def codes(r, kind="blockers"):
    return [f["code"] for f in r[kind]]


def real_article(symbol: str, headline_starts: str) -> dict:
    """A REAL stored article (`news/<symbol>.txt`) as the Data Clerk's fresh hit: its own text, headline, publisher and tickers."""
    doc = next(d for d in load_news(REPO_ROOT / "news" / f"{symbol}.txt") if d.metadata["headline"].startswith(headline_starts))
    md = doc.metadata
    return {"chunk_id": f"{md['article_id']}#0", "article_id": md["article_id"], "chunk": 1, "chunks": 1, "headline": md["headline"], "publisher": md["publisher"], "published_utc": md["published_utc"],
            "url": md.get("url"), "tickers": [symbol], "text": doc.page_content, "score": 0.5, "corpus": "news"}


# ---------------------------------------------------------------- C2: a newer article that looks bad informs, it does not deny


def test_a_newer_article_that_reports_a_blocking_event_is_still_a_blocker_and_denies_with_a_bad_sign_in_another_group():
    """A REAL stored Procter & Gamble article whose sentence code reads as `earnings_miss` (a class in `evidence.blocking_classes`), dated inside the ten days the rule looks back: still the blocker `fresh_negative_event`, the model's
    label notwithstanding. AMENDED (Critic v3, rule V5): it is a sign of the group `news` and denied alone; now ONE bad group approves (the sentence names it, the blocker is shown as a warning because it did not deny), and with a
    second bad group (the same record made a trend one whose stock fell over five sessions: `trend_not_confirmed`, performance) the review is DENIED, capped, with both groups in its reasons."""
    article = {**real_article("PG", "Procter & Gamble Q4 Earnings Beat on Productivity"), "published_utc": "2026-11-04T14:00:00Z"}       # the real text; its date moved to the day before the review so that it is "newer than what the Advisor saw"
    assert ev.bearish_classes(article["text"], "PG", "Procter & Gamble") == ["earnings_miss"] and "earnings_miss" in CONFIG.evidence.blocking_classes
    r = review(captured("PG_open"), fresh={"PG": [article]}, own={})["PG"]
    assert r["approved"] and r["blockers"] == [] and r["groups"]["news"] == ["fresh_negative_event"] and r["requires_human"] is False and r["score"] == r["score_before_blockers"] > 49
    assert "fresh_negative_event" in [w["code"] for w in r["warnings"]] and "earnings_miss" in next(w for w in r["warnings"] if w["code"] == "fresh_negative_event")["message"]
    assert "Note: one reason against it: news (fresh_negative_event:" in r["summary"] and "fresh_negative_news" not in [w["code"] for w in r["warnings"]]
    both = review(tampered_order("PG_open", **{"trade.family": "trend"}), fresh={"PG": [article]}, own={}, changes={"PG": {"ret_5d": -0.021}})["PG"]
    assert not both["approved"] and both["denial_reasons"] == ["group:news", "group:performance", "trend_not_confirmed", "fresh_negative_event"] and both["score"] <= 49 < both["score_before_blockers"]
    assert [b["code"] for b in both["blockers"]] == ["trend_not_confirmed", "fresh_negative_event"] and both["requires_human"] is False
    assert both["summary"].startswith("DENY PG: poor news (fresh_negative_event:") and " and poor performance (trend_not_confirmed:" in both["summary"]


# ---------------------------------------------------------------- C4: a company is the same company whatever its spelling

def real_identities() -> list[tuple[str, str, str]]:
    """Every (symbol, the company the record calls it, the company the symbol directory lists) of the real reviews of the Time Machine's research and of the oil tests' saved runs: what Finnhub's directory really answered."""
    found = {}
    for path in REAL_CACHES:
        order = json.loads(Path(path).read_text(encoding="utf-8")).get("order")
        for r in (order or {}).get("reviews", []):
            idt = (r.get("accuracy") or {}).get("identity") or {}
            if idt.get("listed") and idt.get("listed_company") and idt.get("company"):
                found[(r["symbol"], idt["company"], idt["listed_company"])] = 1
    return list(found)


def test_the_real_names_that_were_called_a_mismatch_are_the_same_company_and_a_different_company_is_not():
    """Rule C4. The real pairs the old rule called `company_mismatch` ('Exxon Mobil' against 'Exxonmobil Holdings Corp', 'APA Corporation' against 'Apa Corp', 'AMD' against 'Advanced Micro Devices'...) are the same company; no pair the old rule
    accepted is lost; and one company's name against another's is still a mismatch."""
    from prometheus.agents.advisor.advisor import names_company
    pairs = real_identities()
    assert {"XOM", "APA"} <= {p[0] for p in pairs}
    for symbol, company, listed in pairs:
        assert names.same_company(symbol, company, listed), (symbol, company, listed)
        assert names_company(listed, symbol, company) <= names.same_company(symbol, company, listed)                  # what the old rule accepted is still accepted
    assert names.same_company("XOM", "Exxon Mobil", "Exxonmobil Holdings Corp") and names.same_company("APA", "APA Corporation", "Apa Corp") and names.same_company("GOOGL", "Alphabet Inc", "Alphabet Inc-Cl A")
    xom, apa = next(p for p in pairs if p[0] == "XOM"), next(p for p in pairs if p[0] == "APA")
    assert not names.same_company("XOM", xom[1], apa[2]) and not names.same_company("APA", apa[1], xom[2]) and not names.same_company("PG", "Procter & Gamble Co", "ACME ROCKET SKATES LTD")
    assert not names.same_company("F", "Ford Motor", "Frontier Group") and not names.same_company("XOM", "", "Exxonmobil Holdings Corp")      # one letter never names a company; no name matches nothing


# ---------------------------------------------------------------- C5: a trend recommendation must be confirmed by the last five sessions


def test_when_the_five_session_return_cannot_be_had_it_is_a_warning_and_not_a_denial_and_it_changes_no_accuracy_number():
    order = tampered_order("PG_open", **{"trade.family": "trend"})
    clerk = FakeClerk(order)
    for results in clerk.metrics.values():
        results.pop("ret_5d")
    r = review(order, clerk=clerk, own={})["PG"]
    assert r["approved"] and "trend_unconfirmed_unknown" in codes(r, "warnings") and r["blockers"] == []
    base = review(tampered_order("PG_open", **{"trade.family": "trend"}), own={})["PG"]            # the same record WITH its ret_5d: asked for and never compared, so the accuracy numbers are the same
    assert r["accuracy"]["match_rate"] == base["accuracy"]["match_rate"] and [x["name"] for x in r["accuracy"]["rechecked"]] == [x["name"] for x in base["accuracy"]["rechecked"]] and "ret_5d" not in [x["name"] for x in base["accuracy"]["rechecked"]]
    clerk2 = FakeClerk(captured("PG_open"))
    review(captured("PG_open"), clerk=clerk2, own={})
    assert "ret_5d" in next(a for a in clerk2.asked if a[0] == "metrics")[2]


# ---------------------------------------------------------------- C6: the same quote gets the same label

def run_with(store, model, **options):
    s = Setup(**options)
    verification = s.verification
    checks, _, _ = rc.check_sources(s.rec, s.claims, s.order, verification, True)
    run = {"stage": "review", "tokens_in": 0, "tokens_out": 0, "calls": 0, "dropped": [], "errors": [], "records_called": 0, "controls_failed": 0}
    result = ev.assess(model, s.rec, s.claims, checks, verification, s.order, ADVISOR_CONFIG, CONFIG, NOW, getattr(model, "model_name", None), True, run, store=store)
    return result, run


def test_another_model_another_prompt_another_source_text_or_a_failed_control_asks_again(tmp_path):
    store = LabelStore(tmp_path / "labels")
    run_with(store, ReviewerModel())
    other_model = ReviewerModel()
    other_model.model_name = "z-ai/glm-5.4"
    run_with(store, other_model)
    assert len(other_model.seen) == 1                                                                                # another model id: asked again
    changed = Setup()
    changed_model = ReviewerModel()
    for c in changed.claims:
        if c.chunk_id:
            changed.verification["chunks"][c.chunk_id]["text"] += " One more sentence that changes the text the model reads."
    checks, _, _ = rc.check_sources(changed.rec, changed.claims, changed.order, changed.verification, True)
    run = {"stage": "review", "tokens_in": 0, "tokens_out": 0, "calls": 0, "dropped": [], "errors": [], "records_called": 0, "controls_failed": 0}
    out = ev.assess(changed_model, changed.rec, changed.claims, checks, changed.verification, changed.order, ADVISOR_CONFIG, CONFIG, NOW, "z-ai/glm-5.3", True, run, store=store)
    assert len(changed_model.seen) == 1 and run["labels_from_store"] < len(out.detail.claims)                        # another source text: the claims whose text changed are asked again (the filing items are not)
    assert prompt_version(25) != prompt_version(24) and prompt_version(25) == prompt_version(25)                     # the prompt's version is its own text: a changed prompt asks again
    bad = LabelStore(tmp_path / "failed")
    liar = ReviewerModel(control="supports")
    r, _ = run_with(bad, liar)
    assert "control_failed" in [f.code for f in r.findings] and bad.count() == 0                                     # a reply that did not read the text keeps nothing
    again = ReviewerModel()
    run_with(bad, again)
    assert len(again.seen) == 1 and bad.count() > 0                                                                 # so it asks again, and the right reply is kept


def test_a_fresh_sources_label_is_kept_too_and_the_review_through_a_critic_with_a_store_makes_one_call_then_none(tmp_path):
    """Through the whole review: the real PG_open record and a real stored article, reviewed by two Critics that share a store. The first calls the model once; the second calls it no more, and its audit says how many labels the store answered."""
    article = real_article("PG", "Procter & Gamble's Turnaround Gains Steam")
    quote = "TD Cowen raises Procter & Gamble's price target to $166 after Q1 beat, but maintains a Hold on rising commodity costs."
    store = LabelStore(tmp_path / "labels")
    reviews = []
    models = []
    for _ in range(2):
        order = captured("PG_open")
        model = ReviewerModel(fresh={"F1": ("bearish", quote)})
        models.append(model)
        critic = Critic(model=model, clerk=FakeClerk(order, fresh={"PG": [article]}, own={}), ledger=MemoryLedger(), label_store=store)
        critic.review_recommendations(order, view(), NOW)
        reviews.append(order.reviews[0])
    assert len(models[0].seen) == 1 and models[1].seen == []
    assert reviews[0]["audit"]["run"]["labels_from_store"] == 0 and reviews[0]["audit"]["run"]["calls"] == 1
    assert reviews[1]["audit"]["run"]["labels_from_store"] == len(reviews[1]["evidence"]["claims"]) + 1 and reviews[1]["audit"]["run"]["calls"] == 0
    assert reviews[1]["evidence"]["fresh"][0]["label"] == "bearish" and reviews[1]["approved"] == reviews[0]["approved"] and reviews[1]["score"] == reviews[0]["score"]
    assert [c["label"] for c in reviews[1]["evidence"]["claims"]] == [c["label"] for c in reviews[0]["evidence"]["claims"]]


def test_the_critics_store_is_the_configs_folder_and_a_test_critic_keeps_none():
    assert CONFIG.evidence.label_store == "data/critic/labels" and open_store(None) is None and open_store(CONFIG.evidence.label_store).folder == REPO_ROOT / "data" / "critic" / "labels"
    assert Critic(model=object()).label_store is None                                                                    # conftest turns the config's store off: no test reads or writes the project's own labels
    assert Critic(model=object(), label_store=None).label_store is None
    assert (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").count("data/critic/") == 1


# ---------------------------------------------------------------- Critic v3: what the Account Manager tells the user about a denial for two reasons

def test_the_account_manager_tells_the_user_a_denial_for_two_groups_in_words_and_a_denial_for_a_fault_as_before():
    """`decisions.denied_lines` reads the review's `denial_reasons`: a denial for bad signs in two groups says "two reasons against it" and then the groups in words with the heaviest code of each (Critic v3, rule V9); a denial for a
    fault (the market is closed) is its one sentence, as it was."""
    from critic_helpers import make_critic

    from prometheus.agents.account_manager import decisions
    last = rec_of(captured("PG_open")).features["last_close"]["value"]
    order = tampered_order("PG_open", **{"trade.family": "trend"})
    make_critic(order, own={}, changes={"PG": {"ret_5d": -0.021, "sma_126": last / 1.5}}).review_recommendations(order, view(), NOW)
    [header, line] = decisions.denied_lines(order)
    assert header == "      DENIED by the Critic, not traded:" and not order.reviews[0]["approved"]
    assert line.startswith("        PG     score 49 (F): two reasons against it, poor performance (trend_not_confirmed: The strategy is trend") and " and a price that is too high (overpriced: The price 167.10 is 50.0% above" in line
    closed = captured("PG_open")
    make_critic(closed, own={}).review_recommendations(closed, view(), at_saturday())
    [_, said] = decisions.denied_lines(closed)
    assert "reasons against it" not in said and said.startswith("        PG     score 49 (F): The market is closed")


def at_saturday():
    from critic_helpers import at
    return at("2026-11-07 12:00")


