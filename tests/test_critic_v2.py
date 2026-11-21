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

from prometheus.agents.advisor.record import Recommendation
from prometheus.agents.critic import chart, names, recheck as rc
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

def test_a_newer_article_that_looks_bad_is_a_warning_that_asks_for_nobody_and_the_review_is_approved():
    """The real PG_open review at 2026-11-05 10:30 with a REAL stored Procter & Gamble article that the model calls bearish (a scripted reply, with a sentence copied from the article) and in which code finds no bad class:
    `fresh_negative_news` was a warning that asked for a person, a DENIAL since entry 103 (it fired 76 times in four six-month tests and 60 of the 72 that closed were winners). Now it informs: the review is approved, carries
    the warning and the lower 'clean' share of the evidence score, and nobody is asked."""
    article = real_article("PG", "Procter & Gamble's Turnaround Gains Steam")
    quote = "TD Cowen raises Procter & Gamble's price target to $166 after Q1 beat, but maintains a Hold on rising commodity costs."
    assert quote in article["text"] and ev.bearish_classes(article["text"], "PG", "Procter & Gamble") == []
    quiet = review(captured("PG_open"), own={})["PG"]
    r = review(captured("PG_open"), model=ReviewerModel(fresh={"F1": ("bearish", quote)}), fresh={"PG": [article]}, own={})["PG"]
    [warn] = [w for w in r["warnings"] if w["code"] == "fresh_negative_news"]
    assert warn["requires_human"] is False and r["requires_human"] is False and r["requires_human_reasons"] == []
    assert r["approved"] and r["denial_reasons"] == [] and r["blockers"] == [] and r["valid_until_utc"]
    assert r["evidence"]["fresh"][0]["label"] == "bearish" and r["dimensions"]["evidence"]["score"] < quiet["dimensions"]["evidence"]["score"] and r["score"] < quiet["score"]      # it still costs evidence points
    assert "fresh_negative_news" not in [w["code"] for w in quiet["warnings"]]


def test_a_newer_article_that_reports_a_blocking_event_still_denies():
    """A REAL stored Procter & Gamble article whose sentence code reads as `earnings_miss` (a class in `evidence.blocking_classes`), dated inside the ten days the rule looks back: still a blocker, the model's label notwithstanding."""
    article = {**real_article("PG", "Procter & Gamble Q4 Earnings Beat on Productivity"), "published_utc": "2026-11-04T14:00:00Z"}       # the real text; its date moved to the day before the review so that it is "newer than what the Advisor saw"
    assert ev.bearish_classes(article["text"], "PG", "Procter & Gamble") == ["earnings_miss"] and "earnings_miss" in CONFIG.evidence.blocking_classes
    r = review(captured("PG_open"), fresh={"PG": [article]}, own={})["PG"]
    assert not r["approved"] and "fresh_negative_event" in codes(r) and "earnings_miss" in r["blockers"][0]["message"] and r["requires_human"] is False and r["score"] <= 49
    assert "fresh_negative_news" not in [w["code"] for w in r["warnings"]]


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


def test_a_real_record_called_exxon_mobil_is_not_a_company_mismatch_but_a_different_company_still_is():
    """The cross-check of the Critic with the directory's real answer: the record calls XOM 'Exxon Mobil', the directory lists 'Exxonmobil Holdings Corp'. It was the warning that asks for a person (a denial); now the identity check passes."""
    order = captured("PG_open")
    rec = rec_of(order).model_copy(update={"symbol": "XOM", "company": "Exxon Mobil"})
    checks, findings, identity, _ = rc.cross_checks(rec, None, None, {"symbols": {"XOM": {"listed": True, "company": "Exxonmobil Holdings Corp", "note": None}}}, CONFIG.accuracy)
    assert findings == [] and next(c for c in checks if c.name == "identity").passed and identity.listed_company == "Exxonmobil Holdings Corp"
    checks, findings, _, _ = rc.cross_checks(rec, None, None, {"symbols": {"XOM": {"listed": True, "company": "Apa Corp", "note": None}}}, CONFIG.accuracy)
    assert [(f.code, f.requires_human) for f in findings] == [("company_mismatch", True)] and "corporate words" in next(c for c in checks if c.name == "identity").note


# ---------------------------------------------------------------- C5: a trend recommendation must be confirmed by the last five sessions

def real_records() -> list[Recommendation]:
    found, seen = [], set()
    for path in REAL_CACHES:
        for rec in (json.loads(Path(path).read_text(encoding="utf-8")).get("order") or {}).get("recommendations", []):
            if rec["identity"]["record_id"] not in seen:
                seen.add(rec["identity"]["record_id"])
                found.append(Recommendation.model_validate(rec))
    return found


def test_over_the_real_records_of_the_oil_weeks_exactly_the_trend_ones_that_fell_are_denied():
    """Rule C5 on the REAL records the Advisor made at four decision weeks, with each one's own return over the five sessions before its decision day (`ret_5d`, which the Advisor and the Critic calculate from the same stored bars):
    `trend_not_confirmed` for a trend record at or under 0.0, nothing for a trend record above it, and nothing for ANY other family however far it fell."""
    records = real_records()
    trend = [r for r in records if r.trade.family == "trend"]
    falling = [r for r in trend if r.features["ret_5d"]["value"] <= CONFIG.chart.trend_confirm_ret_5d]
    others_down = [r for r in records if r.trade.family != "trend" and r.features["ret_5d"]["value"] <= 0]
    assert falling and len(falling) < len(trend) and others_down
    for rec in records:
        found = chart.trend_confirmation(rec, rec.features["ret_5d"]["value"], CONFIG.chart)
        assert [f.code for f in found] == (["trend_not_confirmed"] if rec in falling else []), (rec.symbol, rec.trade.family, rec.features["ret_5d"]["value"])
    only = chart.trend_confirmation(falling[0], falling[0].features["ret_5d"]["value"], CONFIG.chart)[0]
    assert only.severity == "blocker" and only.dimension == "chart" and f"{falling[0].features['ret_5d']['value']:+.1%}" in only.message and only.value == falling[0].features["ret_5d"]["value"] and only.threshold == 0.0


def test_a_trend_review_is_denied_when_the_stock_fell_over_five_sessions_and_the_same_record_is_not_on_a_rising_week():
    """The whole review of the real PG_open record made a TREND recommendation: with the Critic's own re-asked `ret_5d` at -2.1% it is denied for `trend_not_confirmed` (its message says the number, the score is capped), with the real
    +4.1% the record had it is approved, and a CONVICTION record is not touched by the same -2.1%."""
    falling = review(tampered_order("PG_open", **{"trade.family": "trend"}), own={}, changes={"PG": {"ret_5d": -0.021}})["PG"]
    assert not falling["approved"] and codes(falling) == ["trend_not_confirmed"] and falling["score"] <= 49 < falling["score_before_blockers"] and "-2.1%" in falling["blockers"][0]["message"]
    assert falling["summary"].startswith("DENY PG: The strategy is trend") and falling["requires_human"] is False
    rising = review(tampered_order("PG_open", **{"trade.family": "trend"}), own={})["PG"]            # the real +4.05% of the capture
    assert rising["approved"] and codes(rising) == [] and "trend_not_confirmed" not in codes(rising, "warnings") and "trend_unconfirmed_unknown" not in codes(rising, "warnings")
    conviction = review(captured("PG_open"), own={}, changes={"PG": {"ret_5d": -0.021}})["PG"]
    assert conviction["approved"] and codes(conviction) == []


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


def test_the_same_claim_twice_asks_the_model_once_and_the_audit_counts_the_labels_the_store_answered(tmp_path):
    store = LabelStore(tmp_path / "labels")
    first = ReviewerModel(labels={"[5]": "weak"})
    a, run_a = run_with(store, first)
    assert len(first.seen) == 1 and run_a["labels_from_store"] == 0 and a.detail.control.passed and store.count() == len([c for c in a.detail.claims]) + 0
    second = ReviewerModel(default="contradicts")             # a model that would say something else: it must not be asked
    b, run_b = run_with(store, second)
    assert second.seen == [] and run_b["labels_from_store"] == len(b.detail.claims) and run_b["calls"] == 0 and not b.called
    assert [(c.ref, c.label, c.reason) for c in b.detail.claims] == [(c.ref, c.label, c.reason) for c in a.detail.claims] and b.detail.reviewed and b.detail.control is None
    assert b.score == a.score and [f.code for f in b.findings] == [f.code for f in a.findings] and b.detail.counterfactual == a.detail.counterfactual


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


def test_a_store_that_is_off_keeps_nothing_and_a_partly_answered_review_asks_only_for_the_rest(tmp_path):
    first = ReviewerModel()
    run_with(None, first)
    second = ReviewerModel()
    run_with(None, second)
    assert len(first.seen) == len(second.seen) == 1 and not list(tmp_path.glob("*"))
    store = LabelStore(tmp_path / "labels")
    run_with(store, ReviewerModel())
    some = sorted(Path(store.folder).glob("*/*.json"))
    assert len(some) >= 2
    some[0].unlink()                                                                                                 # one label is missing: only that quote is sent (and the control)
    partial = ReviewerModel()
    r, run = run_with(store, partial)
    claims_asked = partial.seen[0][-1].content.split("SOURCES of the claims")[0]
    assert len(partial.seen) == 1 and run["labels_from_store"] == len(r.detail.claims) - 1 and claims_asked.count('"') // 2 == 2 and r.detail.control.passed       # one claim, and the planted control


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
