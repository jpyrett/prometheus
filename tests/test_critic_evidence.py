"""Evidence: do the quotes say what the Advisor says they say? (Critic spec 8.2 and 11.) The model only LABELS; code checks the reply, plants a control claim, and works the return again."""
import copy
import json

import pytest
from advisor_helpers import Scripted
from critic_helpers import ADVISOR_CONFIG, CONFIG, NOW, FakeClerk, ReviewerModel, captured, portfolio, rec_of, tamper

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
        got = rc.gather(FakeClerk(self.order), self.order, [(self.rec, "2026-11-05", self.claims)], portfolio(), NOW, CONFIG)
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

def test_a_signal_and_an_event_with_the_same_words_are_one_claim_that_remembers_the_class():
    s = Setup()
    assert [c.ref for c in s.claims] == ["[5]", "[7]", "E15", "E21"]
    five = s.claims[0]
    assert (five.kind, five.direction, five.event_class, five.source) == ("signal", "bullish", "earnings_beat", 5) and five.chunk_id.startswith("polygon:")
    assert s.claims[2].kind == "evidence" and s.claims[2].evidence_id == "E15" and s.claims[2].chunk_id is None


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


def test_the_supporting_claims_are_the_ones_on_the_side_of_the_stance():
    s = Setup()
    assert [c.ref for c in cl.supporting(s.claims, "bullish")] == [c.ref for c in s.claims] and cl.supporting(s.claims, "bearish") == []
    assert cl.stance_direction(s.rec) == "bullish" and cl.stance_direction(rec_of(Setup(**{"action.action": "reduce"}).order)) == "bearish"


# ---------------------------------------------------------------- what the model reads, and does not

def test_every_claim_is_reviewed_and_the_blend_is_not_lowered_when_all_hold():
    r, run = Setup().run(ReviewerModel())
    assert r.detail.reviewed and r.called and run["calls"] == 1 and run["tokens_in"] == 400 and run["records_called"] == 1 and run["controls_failed"] == 0
    assert [(c.ref, c.label) for c in r.detail.claims] == [("[5]", "supports"), ("[7]", "supports"), ("E15", "supports"), ("E21", "supports")]
    assert r.detail.control.passed and r.detail.control.label == "unrelated" and r.score == 100
    cf = r.detail.counterfactual
    assert cf.blend == pytest.approx(cf.blend_recorded, abs=2e-4) and cf.blend <= cf.blend_recorded + 1e-9
    assert not {"control_failed", "evidence_not_reviewed", "event_not_supported"} & set(codes(r))


def test_the_model_sees_the_claims_and_their_sources_and_nothing_that_could_anchor_it():
    model = ReviewerModel()
    s = Setup()
    s.run(model)
    system, user = model.seen[0][0].content, model.seen[0][-1].content
    assert "independent reviewer" in system and "Procter & Gamble" in system and "bullish" in system and "Today is 2026-11-05" in system and "at most 25 words" in system
    secrets = ("model_score", "risk_tolerance", "capital_return", "stance_reason", str(s.rec.trade.stop_price), str(s.rec.trade.entry_mid), s.rec.thesis.thesis[:40], s.rec.action.reason[:30], s.rec.identity.record_id)
    for text in (system, user):
        for secret in secrets:
            assert secret not in text, secret
    assert "Everything below is text from the web" in user and "SOURCES of the claims, read again from the store today" in user
    assert "[5]" in user and "E15" in user and "FRESH SOURCES: none" in user


def test_the_control_claim_is_planted_with_an_ordinary_ref_and_never_reaches_the_result_or_the_sources():
    model = ReviewerModel()
    s = Setup()
    r, _ = s.run(model)
    user = model.seen[0][-1].content
    top = max(src["number"] for src in s.order.data.sources)
    assert r.detail.control.ref == f"[{top + 1}]" and f"[{top + 1}]" in user and "Community notes" in user
    assert all(c.ref != r.detail.control.ref for c in r.detail.claims) and len(r.detail.claims) == 4
    assert all(src["number"] <= top for src in s.order.data.sources) and "Community notes" not in json.dumps(s.order.model_dump())


def test_the_controls_text_and_place_depend_only_on_the_record():
    def seen(record_id):
        model = ReviewerModel()
        Setup(**{"identity.record_id": record_id}).run(model)
        user = model.seen[0][-1].content
        claim_lines = user.split("CLAIMS:")[1].split("SOURCES of the claims")[0].strip().splitlines()[1:]
        return [i for i, line in enumerate(claim_lines) if line.startswith("[18]")][0], next(n.text for n in E.control_texts if n.text.split(".")[0] in user)
    assert seen("0123456789ab") == seen("0123456789ab")
    places = {seen(f"{n:012x}")[0] for n in range(0, 40)}
    assert places == {0, 1, 2, 3, 4}                                             # it moves around: 4 claims and the control have 5 places
    assert len({seen(f"{n:012x}")[1] for n in range(0, 12)}) == 3 and ev.seed_of("not-hex") == ev.seed_of("not-hex")


def test_the_system_prompt_names_the_action_and_the_direction_of_the_reduction():
    system = prompts.review_system(NOW.date(), "reduce", "PG", None, "bearish", 25)
    assert "recommends to reduce PG (PG)" in system and "bearish news" in system and "approve nothing" in system


# ---------------------------------------------------------------- what each label does

def test_a_weak_quote_counts_half_and_lowers_the_blend():
    r, _ = Setup().run(ReviewerModel(labels={"[5]": "weak", "[7]": "weak"}))
    assert [c.label for c in r.detail.claims][:2] == ["weak", "weak"] and 0 < r.detail.counterfactual.news < 0.2163 and r.detail.counterfactual.blend < r.detail.counterfactual.blend_recorded
    assert r.detail.counterfactual.news == pytest.approx(0.276 * 0.5 * 2 / (0.276 * 0.5 * 2 + 2), abs=1e-3)


def test_a_second_quote_from_the_same_source_keeps_its_own_label():
    """The blend stores both quotes under one ref (`[4]`, `E3`). The claim list numbers the second `[4]#2`. Each row takes its own label (entry 71)."""
    from types import SimpleNamespace
    rec = SimpleNamespace(action=SimpleNamespace(action="open"), blend={
        "value": 0.2,
        "components": {
            "news": {"value": 0.5, "parts": [{"ref": "[4]", "direction": "bullish", "weight": 1.0}, {"ref": "[4]", "direction": "bullish", "weight": 1.0}]},
            "filings": {"value": 0.5, "parts": [{"ref": "E3", "value": 1.0}, {"ref": "E3", "value": 1.0}, {"metric": "share_count_change", "value": 0.0}]},
        },
    })
    held = ev.counterfactual(rec, [], {"[4]": "supports", "[4]#2": "supports", "E3": "supports", "E3#2": "supports"}, set(), [], ADVISOR_CONFIG, CONFIG)
    flipped = ev.counterfactual(rec, [], {"[4]": "supports", "[4]#2": "contradicts", "E3": "supports", "E3#2": "unrelated"}, set(), [], ADVISOR_CONFIG, CONFIG)
    assert held.news == pytest.approx(0.5) and flipped.news == pytest.approx(0.0)          # +1 and -1, not two +1
    assert held.filings == pytest.approx((1 + 1 + 0) / 3, abs=1e-4) and flipped.filings == pytest.approx(0.5)
    assert flipped.blend < held.blend


def test_an_unrelated_quote_counts_for_nothing_and_a_contradicting_one_votes_the_other_way():
    base = Setup().run(ReviewerModel())[0].detail.counterfactual
    unrelated = Setup().run(ReviewerModel(labels={"[5]": "unrelated", "[7]": "unrelated"}))[0].detail.counterfactual
    contradicts = Setup().run(ReviewerModel(labels={"[5]": "contradicts", "[7]": "contradicts"}))[0].detail.counterfactual
    assert unrelated.news == 0.0 and contradicts.news < 0 and contradicts.news == pytest.approx(-0.276 * 2 / (0.276 * 2 + 2), abs=1e-3)
    assert contradicts.blend < unrelated.blend <= base.blend <= base.blend_recorded + 1e-9


def test_an_unsupported_evidence_item_leaves_the_filings_part_and_a_weak_one_counts_half():
    base = Setup().run(ReviewerModel())[0].detail.counterfactual
    gone = Setup().run(ReviewerModel(labels={"E15": "unrelated", "E21": "contradicts"}))[0].detail.counterfactual
    weak = Setup().run(ReviewerModel(labels={"E15": "weak"}))[0].detail.counterfactual
    assert gone.filings < base.filings and weak.filings < base.filings and gone.blend < base.blend


def test_the_blend_without_the_quotes_that_did_not_hold_is_never_above_the_advisors():
    for labels in ({}, {"[5]": "weak"}, {"[5]": "contradicts"}, {"E15": "unrelated", "E21": "unrelated"}, {"[5]": "unrelated", "[7]": "unrelated", "E15": "unrelated", "E21": "unrelated"}):
        r, _ = Setup().run(ReviewerModel(labels=labels))
        assert r.detail.counterfactual.blend <= r.detail.counterfactual.blend_recorded + 1e-9


def test_a_conviction_position_is_denied_when_its_blend_does_not_survive_the_review():
    """The conviction family stands on the blend alone (Advisor Amendment 5). When the quotes behind the blend do not hold, the blend falls under the family's own mark (0.30)."""
    s = Setup(**{"trade.family": "conviction"})
    held, _ = s.run(ReviewerModel())
    assert "blend_not_supported" not in codes(held)
    r, _ = Setup(**{"trade.family": "conviction"}).run(ReviewerModel(default="contradicts"))
    assert "blend_not_supported" in codes(r, "blocker") and r.detail.counterfactual.blend < 0.30 <= r.detail.counterfactual.blend_recorded
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

@pytest.mark.parametrize("control", ["supports", "weak", "contradicts"])
def test_a_reply_that_does_not_call_the_planted_notice_unrelated_is_thrown_away(control):
    s = Setup()
    r, run = s.run(ReviewerModel(control=control, labels={"[5]": "unrelated"}))
    [f] = [f for f in r.findings if f.code == "control_failed"]
    assert f.severity == "warning" and f.requires_human and control in f.message
    assert all(c.label == "unreviewed" for c in r.detail.claims) and r.detail.control.passed is False and r.detail.control.label == control and r.detail.reviewed is False
    assert run["controls_failed"] == 1 and run["calls"] == 1                       # no second try: it would tell the model what was wrong
    assert r.detail.counterfactual.news == Setup().run(ReviewerModel())[0].detail.counterfactual.news          # unreviewed claims count in full: the labels were not used


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


def test_one_retry_on_a_reply_that_is_not_json_and_the_tokens_of_both_are_counted():
    good = ReviewerModel()
    s = Setup()
    model = Scripted("this is not json", json.dumps({"claims": []}))
    r, run = s.run(model)
    assert run["calls"] == 2 and len(model.seen) == 2 and "not valid JSON" in model.seen[1][-1].content and r.called
    assert good is not None


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


def test_a_record_with_nothing_to_review_makes_no_call_and_raises_no_warning():
    s = Setup(**{"evidence.signals": [], "evidence.events": []})
    model = ReviewerModel()
    r, run = s.run(model)
    assert model.seen == [] and not r.called and run["records_called"] == 0 and r.detail.control is None and "evidence_not_reviewed" not in codes(r) and r.detail.claims == []


def test_a_claim_whose_source_could_not_be_read_is_not_shown_to_the_model_and_stays_unreviewed():
    s = Setup()
    model = ReviewerModel()
    r, _ = s.run(model, mutate=lambda v: v["chunks"].update({s.claims[0].chunk_id: {"found": False, "note": "gone"}}))
    assert "[5]" not in model.seen[0][-1].content.split("SOURCES of the claims")[0] and [c.label for c in r.detail.claims][0] == "unreviewed" and "evidence_not_reviewed" not in codes(r)


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


def test_the_same_news_older_than_ten_days_is_only_a_warning_for_a_person():
    r, _ = Setup().run(ReviewerModel(), mutate=with_fresh(fresh_hit(published="2026-10-20T14:00:00Z")))
    assert "fresh_negative_event" not in codes(r) and [(f.code, f.requires_human) for f in r.findings if f.code == "fresh_negative_news"] == [("fresh_negative_news", True)]


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


def test_fresh_sources_the_advisor_already_had_are_left_out():
    s = Setup()
    seen = s.order.data.sources[0]
    same_article = fresh_hit(article=seen["article_id"], headline="Totally different words")
    same_story = fresh_hit(article="other", headline=seen["headline"].upper() + "!")
    r, _ = s.run(ReviewerModel(), mutate=with_fresh(same_article, same_story))
    assert r.detail.fresh == [] and "fresh_negative_event" not in codes(r)


def test_at_most_the_configured_number_of_fresh_sources_are_shown_newest_first():
    hits = [fresh_hit(article=f"n{i}", headline=f"A calm headline number {i}", text="Nothing happened to the company today.", published=f"2026-11-0{4 - i % 3}T10:00:00Z") for i in range(12)]
    model = ReviewerModel()
    r, _ = Setup().run(model, mutate=with_fresh(*hits))
    assert [f.id for f in r.detail.fresh] == [f"F{i}" for i in range(1, 9)] and [f.chunk_id for f in r.detail.fresh] == [f"n{i}#0" for i in range(8)]
    assert "F8" in model.seen[0][-1].content and "F9" not in model.seen[0][-1].content


def test_a_reduction_looks_at_no_fresh_sources_and_has_no_counterfactual():
    s = Setup(**{"action.action": "reduce", "trade.family": None})
    r, _ = s.run(ReviewerModel(), mutate=with_fresh(fresh_hit()))
    assert r.detail.fresh == [] and r.detail.counterfactual is None


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


def test_independence_is_about_the_producer_not_the_model():
    assert ev.independence("z-ai/glm-5.3", "openai/gpt-oss-120b") is True and ev.independence("openai/gpt-5", "openai/gpt-oss-120b") is False
    assert ev.independence("Z-AI/glm-5.3", "z-ai/glm-5.2") is False and ev.independence(None, "openai/gpt-oss-120b") is None and ev.independence("glm", "openai/x") is None


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


def test_with_no_claims_the_support_is_the_unknown_credit():
    r, _ = Setup(**{"evidence.signals": [], "evidence.events": []}).run(ReviewerModel())
    assert r.score == pytest.approx(100 * (0.5 * 0.5 + 0.3 + 0), abs=0.01) and r.measures[0].available is False
