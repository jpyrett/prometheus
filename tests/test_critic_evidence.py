"""Evidence: do the quotes say what the Advisor says they say? (Critic spec 8.2 and 11.) The model only LABELS; code checks the reply, plants a control claim, and works the return again."""
import copy
import json

import pytest
from advisor_helpers import Scripted
from critic_helpers import ADVISOR_CONFIG, CONFIG, NOW, FakeClerk, ReviewerModel, captured, rec_of, tamper

from prometheus.agents.critic import claims as cl
from prometheus.agents.critic import evidence as ev
from prometheus.agents.critic import recheck as rc
from prometheus.prompts import critic as prompts

E = CONFIG.evidence


def new_run():
    return {"stage": "review", "tokens_in": 0, "tokens_out": 0, "calls": 0, "dropped": [], "errors": [], "records_called": 0, "controls_failed": 0}


class Setup:
    """The captured PG_open order, its claims, their sources as the Data Clerk reads them again, and a way to run the evidence step on them."""

    def __init__(self, name="PG_open", **changes):
        self.order = captured(name)
        if changes:
            self.order.recommendations[0] = tamper(self.order.recommendations[0], **changes)
        self.rec = rec_of(self.order)
        self.claims = cl.claims_of(self.rec, self.order)
        got = rc.gather(FakeClerk(self.order), self.order, [(self.rec, "2026-11-05", self.claims)], NOW, CONFIG)
        self.verification = copy.deepcopy(got.verification)
        self.verified = got.verified

    def run(self, model, independent=True, critic_model="z-ai/glm-5.3", run=None, mutate=None):
        verification = copy.deepcopy(self.verification)
        if mutate:
            mutate(verification)
        checks, _, _ = rc.check_sources(self.rec, self.claims, self.order, verification, True)
        run = run if run is not None else new_run()
        return ev.assess(model, self.rec, self.claims, checks, verification, self.order, ADVISOR_CONFIG, CONFIG, NOW, critic_model, independent, run), run


def codes(result, severity=None):
    return [f.code for f in result.findings if severity in (None, f.severity)]


def fresh_hit(article="new1", headline="Procter & Gamble lowers its full-year outlook", text="Procter & Gamble lowered its full-year outlook and cut guidance for the fiscal year. Shares fell.",
              published="2026-11-04T14:00:00Z", corpus="news", publisher="Reuters"):
    return {"chunk_id": f"{article}#0", "article_id": article, "chunk": 1, "chunks": 1, "headline": headline, "publisher": publisher, "published_utc": published, "url": "https://x/1", "tickers": ["PG"],
            "text": text, "score": 0.5, "corpus": corpus}


# ---------------------------------------------------------------- the claims


def test_only_the_model_origin_events_and_counted_verified_signals_are_claims():
    order = captured("PG_open")
    rec = rec_of(order)
    order.recommendations[0] = tamper(order.recommendations[0], **{"evidence.signals.0.counted": False, "evidence.events.0.origin": "filing_index"})
    refs = [c.ref for c in cl.claims_of(rec_of(order), order)]
    assert "[5]" not in refs and "earnings_beat" not in refs and "[7]" in refs
    assert rec is not None


def test_two_events_of_one_class_from_different_sources_get_unique_refs():
    order = captured("PG_open")
    for source, quote in ((6, "Procter & Gamble raised its outlook for the year."), (8, "Procter & Gamble lifted its full-year forecast again.")):
        event = copy.deepcopy(order.recommendations[0]["evidence"]["events"][0])
        event.update(source=source, quote=quote, **{"class": "guidance_raise"})
        order.recommendations[0]["evidence"]["events"].append(event)
    refs = [c.ref for c in cl.claims_of(rec_of(order), order)]
    assert refs.count("guidance_raise") == 1 and refs.count("guidance_raise#2") == 1 and len(refs) == len(set(refs))


# ---------------------------------------------------------------- what the model reads, and does not


# ---------------------------------------------------------------- what each label does


def test_an_unrelated_quote_counts_for_nothing_and_a_contradicting_one_votes_the_other_way():
    base = Setup().run(ReviewerModel())[0].detail.counterfactual
    unrelated = Setup().run(ReviewerModel(labels={"[5]": "unrelated", "[7]": "unrelated"}))[0].detail.counterfactual
    contradicts = Setup().run(ReviewerModel(labels={"[5]": "contradicts", "[7]": "contradicts"}))[0].detail.counterfactual
    assert unrelated.news == 0.0 and contradicts.news < 0 and contradicts.news == pytest.approx(-0.276 * 2 / (0.276 * 2 + 2), abs=1e-3)
    assert contradicts.blend < unrelated.blend <= base.blend <= base.blend_recorded + 1e-9


def test_the_blend_without_the_quotes_that_did_not_hold_is_never_above_the_advisors():
    for labels in ({}, {"[5]": "weak"}, {"[5]": "contradicts"}, {"E15": "unrelated", "E21": "unrelated"}, {"[5]": "unrelated", "[7]": "unrelated", "E15": "unrelated", "E21": "unrelated"}):
        r, _ = Setup().run(ReviewerModel(labels=labels))
        assert r.detail.counterfactual.blend <= r.detail.counterfactual.blend_recorded + 1e-9


def test_a_conviction_positions_blend_that_does_not_survive_the_review_is_a_warning():
    """The conviction family stands on the blend alone (Advisor Amendment 5). When the quotes behind the blend do not hold, the blend falls under the family's own mark (0.30). AMENDED (Critic v2, rule C3): that is the WARNING
    `blend_not_supported` now, no longer a blocker (the blend worked again is still calculated and the evidence score still loses its points)."""
    s = Setup(**{"trade.family": "conviction"})
    held, _ = s.run(ReviewerModel())
    assert "blend_not_supported" not in codes(held)
    r, _ = Setup(**{"trade.family": "conviction"}).run(ReviewerModel(default="contradicts"))
    assert "blend_not_supported" in codes(r, "warning") and "blend_not_supported" not in codes(r, "blocker") and r.detail.counterfactual.blend < 0.30 <= r.detail.counterfactual.blend_recorded
    assert not [f for f in r.findings if f.code == "blend_not_supported"][0].requires_human and r.score < held.score        # it asks for nobody, and the evidence score is lower
    assert "blend_not_supported" not in codes(Setup(**{"trade.family": "capital_return"}).run(ReviewerModel(default="contradicts"))[0])   # opened on another strategy: not judged by the blend alone


def test_when_the_event_that_opened_the_position_is_not_supported_it_is_a_blocker():
    s = Setup(**{"trade.family": "event"})
    r, _ = s.run(ReviewerModel(labels={"[5]": "unrelated"}))
    assert "event_not_supported" in codes(r, "blocker") and "earnings_beat" in r.findings[0].message
    assert "event_not_supported" not in codes(Setup(**{"trade.family": "event"}).run(ReviewerModel())[0])
    assert "event_not_supported" not in codes(Setup().run(ReviewerModel(labels={"[5]": "unrelated"}))[0])      # the position was not opened on the event


def test_a_source_that_does_not_list_the_symbol_counts_for_nothing_in_the_return():
    s = Setup()
    base = s.run(ReviewerModel())[0].detail.counterfactual

    def untagged(v):
        for claim in s.claims:
            if claim.chunk_id:
                v["chunks"][claim.chunk_id]["tickers"] = ["MSFT"]
    r, _ = s.run(ReviewerModel(), mutate=untagged)
    assert r.detail.counterfactual.news == 0.0 and r.detail.counterfactual.blend < base.blend


# ---------------------------------------------------------------- the control claim: proof the model read the text


def test_a_model_that_labels_everything_supports_fails_the_control():
    r, run = Setup().run(Scripted(json.dumps({"claims": [{"ref": f"[{n}]", "label": "supports"} for n in list(range(1, 25))] + [{"ref": "E15", "label": "supports"}, {"ref": "E21", "label": "supports"}], "fresh": []})))
    assert "control_failed" in codes(r) and run["controls_failed"] == 1


def test_a_reply_that_leaves_the_control_out_fails_it_too():
    model = Scripted(json.dumps({"claims": [{"ref": "[5]", "label": "supports"}], "fresh": []}))
    r, _ = Setup().run(model)
    assert r.detail.control.label is None and "control_failed" in codes(r)


def test_a_failed_control_discards_the_models_fresh_labels_but_not_what_code_found():
    s = Setup()
    hit = fresh_hit(published="2026-11-04T14:00:00Z")
    r, _ = s.run(ReviewerModel(control="supports", fresh={"F1": ("bearish", "Procter & Gamble lowered its full-year outlook and cut guidance for the fiscal year.")}),
                 mutate=lambda v: v["fresh"].update(PG=[hit]))
    assert r.detail.fresh[0].label == "none" and r.detail.fresh[0].classes == ["guidance_cut"] and "fresh_negative_event" in codes(r, "blocker")


# ---------------------------------------------------------------- a model that fails, or answers badly

def test_a_model_that_raises_is_not_reviewed_and_says_so_for_a_person():
    r, run = Setup().run(ReviewerModel(raw=TimeoutError("took too long")))
    [f] = [f for f in r.findings if f.code == "evidence_not_reviewed"]
    assert f.requires_human and "TimeoutError" in f.message and r.detail.reviewed is False and all(c.label == "unreviewed" for c in r.detail.claims) and run["errors"] and "TimeoutError" in run["errors"][0]
    assert r.score == pytest.approx(100 * (0.5 * 0.5 + 0.3 + 0.2 * 0), abs=0.01)           # unreviewed: the unknown credit for support, nothing for breadth


def test_a_claim_the_model_leaves_unlabelled_is_unreviewed_and_named():
    class Partial(ReviewerModel):
        def invoke(self, messages):
            reply = json.loads(super().invoke(messages).content)
            reply["claims"] = [c for c in reply["claims"] if c["ref"] != "[7]"]
            return type("R", (), {"content": json.dumps(reply), "usage_metadata": {}})()
    r, _ = Setup().run(Partial())
    [f] = [f for f in r.findings if f.code == "evidence_not_reviewed"]
    assert "[7]" in f.message and f.requires_human and [c.label for c in r.detail.claims if c.ref == "[7]"] == ["unreviewed"] and r.detail.control.passed


def test_labels_for_claims_that_were_not_asked_or_not_on_the_list_are_dropped_and_recorded():
    class Noisy(ReviewerModel):
        def invoke(self, messages):
            reply = json.loads(super().invoke(messages).content)
            reply["claims"] += [{"ref": "[99]", "label": "supports"}, {"ref": "[5]", "label": "wonderful"}, "nonsense", {"label": "supports"}]
            next(c for c in reply["claims"] if c["ref"] == "[5]")["label"] = "SUPPORTS"
            return type("R", (), {"content": json.dumps(reply), "usage_metadata": {}})()
    r, run = Setup().run(Noisy())
    assert r.detail.reviewed and any("[99]" in d for d in run["dropped"]) and any("'wonderful'" in d or "not a label" in d for d in run["dropped"])
    assert {c.ref: c.label for c in r.detail.claims}["[5]"] == "supports"      # the first valid label for a ref wins, whatever its case


def test_no_model_means_not_reviewed_and_no_call():
    r, run = Setup().run(None)
    assert "evidence_not_reviewed" in codes(r) and run["calls"] == 0 and not r.called and "no model" in run["errors"][0]


def test_the_orders_budget_of_model_calls_is_respected():
    model = ReviewerModel()
    r, run = Setup().run(model, run={**new_run(), "records_called": CONFIG.limits.max_model_calls})
    assert model.seen == [] and "evidence_not_reviewed" in codes(r) and "used up" in run["errors"][0]


def test_a_source_longer_than_the_limit_is_shown_around_its_quote():
    text = ("filler " * 3000) + "Procter & Gamble lifted its outlook for the full year. " + ("more filler " * 3000)
    shown = ev.excerpt(text, "Procter & Gamble lifted its outlook for the full year.", 4000)
    assert len(shown) < 4100 and "Procter & Gamble lifted its outlook" in shown and shown.startswith("... ") and shown.endswith(" ...")
    assert ev.excerpt("short text", "short", 4000) == "short text" and len(ev.excerpt("x" * 9000, "", 4000)) <= 4004


# ---------------------------------------------------------------- the fresh sources

def with_fresh(*hits):
    return lambda v: v["fresh"].update(PG=list(hits))


def test_newer_bad_news_that_code_finds_is_a_blocker_whatever_the_model_says():
    r, _ = Setup().run(ReviewerModel(), mutate=with_fresh(fresh_hit()))
    [f] = [f for f in r.findings if f.code == "fresh_negative_event"]
    assert f.severity == "blocker" and "guidance_cut" in f.message and "F1" in f.message and r.detail.fresh[0].classes == ["guidance_cut"] and r.detail.fresh[0].label == "none"
    base = Setup().run(ReviewerModel())[0]
    assert r.detail.counterfactual.news < base.detail.counterfactual.news and r.detail.counterfactual.blend < base.detail.counterfactual.blend and r.score < base.score


# The real sentence that blocked a Procter & Gamble purchase for PepsiCo's news (stored article, 2026-11-13; the first live run of 2026-11-14).
PEPSICO_SENTENCE = ("Procter & Gamble or PepsiCo: The One Dividend Stock I’d Hold for the Next 20 Years Key TakeawaysProcter & Gamble declared its $1. 0885 quarterly dividend in October, "
                    "while PepsiCo cut its 2026 core EPS growth outlook and lost KFC’s U.")


def test_a_cut_by_another_company_in_the_same_sentence_is_not_this_companys_cut():
    assert ev.bearish_classes(PEPSICO_SENTENCE, "PG", "Procter & Gamble") == []                                     # it was ["guidance_cut"] and blocked the purchase
    assert ev.bearish_classes(PEPSICO_SENTENCE, "PEP", "PepsiCo") == ["guidance_cut"]                                # for PepsiCo the same sentence is a real finding
    for joiner in (", while ", ", whereas ", ", but ", "; "):
        text = f"Procter & Gamble declared its quarterly dividend{joiner}PepsiCo cut its 2026 guidance."
        assert ev.bearish_classes(text, "PG", "Procter & Gamble") == [], joiner


def test_the_companys_own_cut_still_counts_in_every_shape_of_sentence():
    assert ev.bearish_classes("Procter & Gamble cut its 2026 core EPS growth outlook.", "PG", "Procter & Gamble") == ["guidance_cut"]
    assert ev.bearish_classes("PepsiCo declared a dividend, while Procter & Gamble cut its 2026 guidance.", "PG", "Procter & Gamble") == ["guidance_cut"]
    assert ev.bearish_classes("Procter & Gamble cut its 2026 guidance, while PepsiCo declared a dividend.", "PG", "Procter & Gamble") == ["guidance_cut"]
    assert ev.bearish_classes("Both Procter & Gamble and PepsiCo cut their 2026 guidance.", "PG", "Procter & Gamble") == ["guidance_cut"]     # one clause names both: it is about both


def test_a_review_does_not_block_a_purchase_for_another_companys_news():
    """End to end through the review: the stored article says PepsiCo cut its outlook; the model labelled it `none`; the code no longer finds a class either, so there is no `fresh_negative_event`."""
    hit = fresh_hit(text=PEPSICO_SENTENCE, headline="Procter & Gamble or PepsiCo: The One Dividend Stock I’d Hold for the Next 20 Years")
    r, _ = Setup().run(ReviewerModel(), mutate=with_fresh(hit))
    assert "fresh_negative_event" not in codes(r) and r.detail.fresh[0].classes == [] and r.detail.fresh[0].label == "none"


def test_a_bad_class_that_is_not_in_the_blocking_list_is_a_warning():
    hit = fresh_hit(text="Procter & Gamble announced a Form 144 sale notice... Procter & Gamble reported a restatement of its financial statements.", headline="x")
    classes = ev.bearish_classes(hit["text"], "PG", "Procter & Gamble")
    assert "restatement" in classes and "restatement" in E.blocking_classes
    other = fresh_hit(text="Procter & Gamble's chief executive departed unexpectedly after the board lost confidence.", headline="CEO leaves")
    if ev.bearish_classes(other["text"], "PG", "Procter & Gamble") and not set(ev.bearish_classes(other["text"], "PG", "Procter & Gamble")) & set(E.blocking_classes):
        r, _ = Setup().run(ReviewerModel(), mutate=with_fresh(other))
        assert "fresh_negative_event" not in codes(r) and "fresh_negative_news" in codes(r)


def test_a_bearish_label_with_a_quote_that_is_in_the_text_and_names_the_company_counts():
    quote = "Procter & Gamble said demand for its premium brands is weakening sharply."
    hit = fresh_hit(text=f"Analysts met this week. {quote} The shares were flat.", headline="Demand worry")
    r, run = Setup().run(ReviewerModel(fresh={"F1": ("bearish", quote)}), mutate=with_fresh(hit))
    item = r.detail.fresh[0]
    assert (item.label, item.quote, item.classes) == ("bearish", quote, []) and "fresh_negative_news" in codes(r, "warning") and "fresh_negative_event" not in codes(r) and run["dropped"] == []
    assert r.detail.counterfactual.news < Setup().run(ReviewerModel())[0].detail.counterfactual.news and r.measures[1].value == 0.5        # one bad item halves the "clean" share


@pytest.mark.parametrize("quote", ["Procter & Gamble is doomed and everyone should sell it now.", "Demand is weakening sharply.", None])
def test_a_bearish_label_whose_quote_is_not_in_the_source_or_does_not_name_the_company_is_ignored(quote):
    hit = fresh_hit(text="A calm day. Procter & Gamble said demand for its premium brands is weakening sharply.", headline="x")
    r, run = Setup().run(ReviewerModel(fresh={"F1": ("bearish", quote)}), mutate=with_fresh(hit))
    assert r.detail.fresh[0].label == "none" and "fresh_negative_news" not in codes(r) and any("ignored" in d for d in run["dropped"])


def test_a_reduction_is_not_a_record_so_the_evidence_step_never_sees_one_and_a_purchase_always_reads_the_fresh_sources():
    """AMENDED (entry 105): a `reduce` looked at no fresh sources and had no counterfactual. Such a record cannot be built now, and every record is a purchase: the fresh sources are always read."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="the action is 'reduce': a recommendation is to buy"):
        Setup(**{"action.action": "reduce"})
    r, _ = Setup().run(ReviewerModel(), mutate=with_fresh(fresh_hit()))
    assert r.detail.fresh and r.detail.counterfactual is not None


def test_a_press_release_counts_for_less_than_an_article_in_the_return():
    release = fresh_hit(corpus="press_releases", article="pr:1", publisher="Press release")
    article = fresh_hit(corpus="news")
    a = Setup().run(ReviewerModel(), mutate=with_fresh(article))[0].detail.counterfactual
    p = Setup().run(ReviewerModel(), mutate=with_fresh(release))[0].detail.counterfactual
    assert p.news > a.news and p.blend >= a.blend


# ---------------------------------------------------------------- the findings about the claims themselves

def test_the_same_producer_as_the_advisor_is_a_warning_for_a_person():
    r, _ = Setup().run(ReviewerModel(), independent=False, critic_model="openai/gpt-oss-120b")
    [f] = [f for f in r.findings if f.code == "same_model_family"]
    assert f.requires_human and "openai/gpt-oss-120b" in f.message
    assert "same_model_family" not in codes(Setup().run(ReviewerModel(), independent=None)[0]) and "same_model_family" not in codes(Setup().run(ReviewerModel())[0])


def test_the_newest_quote_being_old_single_publishers_and_counter_evidence_are_noted():
    s = Setup()
    r, _ = s.run(ReviewerModel())
    assert "evidence_old" in codes(r, "warning") and "13 days" in [f for f in r.findings if f.code == "evidence_old"][0].message
    r, _ = s.run(ReviewerModel(), mutate=lambda v: [v["chunks"][c.chunk_id].update(publisher="Yahoo", published_utc="2026-11-04T10:00:00Z") for c in s.claims if c.chunk_id])
    assert "evidence_old" not in codes(r) and "single_publisher" in codes(r, "warning")
    s = Setup(**{"evidence.counter_evidence": [{"ref": "[9]", "direction": "bearish", "quote": "x"}]})
    assert [f.severity for f in s.run(ReviewerModel())[0].findings if f.code == "counter_evidence"] == ["info"]


# ---------------------------------------------------------------- the score

def test_the_evidence_score_is_support_cleanliness_and_breadth():
    weak = Setup().run(ReviewerModel(labels={"[5]": "weak", "[7]": "unrelated", "E15": "supports", "E21": "contradicts"}))[0]
    support = (0.5 + 0.0 + 1.0 + 0.0) / 4
    breadth = min(1, 2 / 3)
    assert weak.score == pytest.approx(100 * (0.5 * support + 0.3 * 1 + 0.2 * breadth), abs=0.01)
    assert [m.name for m in weak.measures] == ["support", "clean", "breadth"] and [m.weight for m in weak.measures] == [50, 30, 20]


