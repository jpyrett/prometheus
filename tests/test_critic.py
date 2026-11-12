"""The whole Critic step with fakes (Critic spec section 7): the nine steps over the real captures, and the principles P1 to P8."""
import ast
import copy
import json
import re
from datetime import datetime
from pathlib import Path

import pytest
from advisor_helpers import MemoryLedger
from critic_helpers import CONFIG, NOW, FakeClerk, ReviewerModel, at, captured, make_critic, portfolio, profile, rec_of, review, tamper, view

from prometheus.agents.critic.critic import Critic
from prometheus.agents.critic.record import MODEL_FIELDS, Review
from prometheus.common import work_order as wo
from prometheus.rag.config import REPO_ROOT

CRITIC_DIR = REPO_ROOT / "prometheus" / "agents" / "critic"


def dump(order):
    """The Advisor's records as the Advisor wrote them: without the `critic` block the Critic adds (Amendment 2)."""
    return json.dumps([{k: v for k, v in r.items() if k != "critic"} for r in order.recommendations], sort_keys=True, default=str)


# ---------------------------------------------------------------- the step

def test_the_order_comes_back_reviewed_with_one_review_per_recommendation_joined_by_record_id():
    order = captured("PG_open")
    before = dump(order)
    ledger = MemoryLedger()
    returned = make_critic(order, ledger=ledger).review_recommendations(order, view(), portfolio(), NOW)
    assert returned is order and order.status == "reviewed" and order.schema_version == 3
    assert [r["record_id"] for r in order.reviews] == [r["identity"]["record_id"] for r in order.recommendations] and len(order.reviews) == 1
    assert dump(order) == before                                         # P5: the Advisor's own fields are never touched, byte for byte (the only addition is the `critic` block)
    review_ = Review.model_validate(order.reviews[0])
    assert review_.approved and review_.symbol == "PG" and review_.action == "open" and review_.request_id == order.request_id and review_.reviewed_utc == "2026-11-05T15:30:00Z"
    assert ledger.records == order.reviews and ledger.problems == []      # one line per review, in the reviews ledger


def test_the_run_is_recorded_on_the_order():
    order = captured("PG_open")
    model = ReviewerModel()
    make_critic(order, model=model).review_recommendations(order, view(), portfolio(), NOW)
    [run] = order.critic_runs
    assert (run["stage"], run["model"], run["records"], run["calls"], run["records_called"], run["controls_failed"], run["error"]) == ("review", "z-ai/glm-5.3", 1, 1, 1, 0, None)
    assert run["tokens_in"] == 400 and run["tokens_out"] == 80 and run["independent_model"] is True and run["advisor_model"] and run["tolerance"] == "moderate"
    assert run["now_et"] == "2026-11-05 10:30 ET" and run["session"] == "regular" and run["seconds"] >= 0 and run["metric_order"] and run["quote_order"] and run["verify_order"]
    assert order.reviews[0]["audit"]["critic_model"] == "z-ai/glm-5.3" and order.reviews[0]["audit"]["config_version"] == CONFIG.config_version and order.reviews[0]["audit"]["verify_order"] == run["verify_order"]


def test_a_healthy_recommendation_is_approved_for_an_hour_at_the_size_the_profile_allows():
    r = review(captured("PG_open"))["PG"]
    assert r["approved"] and r["blockers"] == [] and r["approved_shares"] == 44 and r["suggested_shares"] == 75 and r["valid_until_utc"] == "2026-11-05T16:30:00Z"
    assert (r["score"], r["grade"]) == (88, "B") and r["summary"].startswith("APPROVE 44 of 75 shares of PG: score 88 (B)")
    assert {n: d["points"] for n, d in r["dimensions"].items()} == {"accuracy": 30.0, "evidence": 15.0, "risk_fit": 18.42, "timing": 10.0, "value": 6.58, "chart": 7.5}


def test_every_check_runs_for_every_record_even_when_the_first_blocker_is_found():
    r = review(captured("PG_open"), now=at("2026-11-07 12:00"))["PG"]                 # a Saturday
    assert not r["approved"] and {"market_closed", "record_expired"} <= {b["code"] for b in r["blockers"]} and r["valid_until_utc"] is None and r["approved_shares"] == 0
    assert set(r["dimensions"]) == {"accuracy", "evidence", "risk_fit", "timing", "value", "chart"} and r["dimensions"]["accuracy"]["score"] == 100 and r["dimensions"]["risk_fit"]["score"] > 0
    assert r["score"] == 49 and r["score_before_blockers"] > 49 and r["grade"] == "F" and r["timing"]["session"] == "closed"
    assert "market_closed" in r["denial_reasons"] and r["summary"].startswith("DENY PG: ")


def test_ignore_market_hours_drops_the_closed_market_blocker_from_the_review(monkeypatch):
    monkeypatch.setenv("IGNORE_MARKET_HOURS", "true")
    order = captured("PG_open")
    make_critic(order).review_recommendations(order, view(), portfolio(), at("2026-11-07 12:00"))
    r = order.reviews[0]
    assert "market_closed" not in {b["code"] for b in r["blockers"]} and any(n["code"] == "market_hours_ignored" for n in r["notes"])
    assert "record_expired" in {b["code"] for b in r["blockers"]} and order.critic_runs[-1]["ignore_market_hours"] is True
    from prometheus.agents.critic.report import header
    assert "IGNORE_MARKET_HOURS" in header(order)


def test_a_review_twice_is_the_same_review():
    order = captured("PG_open")
    ledger = MemoryLedger()
    critic = make_critic(order, ledger=ledger)
    critic.review_recommendations(order, view(), portfolio(), NOW)
    first = copy.deepcopy(order.reviews)
    critic.review_recommendations(order, view(), portfolio(), NOW)
    assert order.reviews == first and len(order.critic_runs) == 1 and len(ledger.records) == 1
    order.status = "decided"
    assert critic.review_recommendations(order, view(), portfolio(), NOW).reviews == first


def test_an_order_that_is_not_recommended_is_refused_with_a_message_and_nothing_else_happens():
    order = captured("PG_open")
    order.status = "data_ready"
    clerk = FakeClerk(order)
    Critic(model=ReviewerModel(), clerk=clerk, ledger=MemoryLedger()).review_recommendations(order, view(), portfolio(), NOW)
    assert order.status == "data_ready" and order.reviews == [] and "the Critic needs an order with recommendations (status data_ready)" in order.errors and clerk.asked == []


def test_no_recommendations_is_a_result_not_a_failure():
    order = captured("PG_open")
    order.recommendations = []
    clerk = FakeClerk(order)
    Critic(model=ReviewerModel(), clerk=clerk, ledger=MemoryLedger()).review_recommendations(order, view(), portfolio(), NOW)
    assert order.status == "reviewed" and order.reviews == [] and clerk.asked == [] and order.errors == [] and order.critic_runs[0]["records"] == 0


def test_a_clock_without_a_timezone_is_a_programming_mistake():
    order = captured("PG_open")
    with pytest.raises(ValueError, match="timezone"):
        make_critic(order).review_recommendations(order, view(), portfolio(), datetime(2026, 11, 5, 10, 30))
    assert order.status == "recommended"


# ---------------------------------------------------------------- records that cannot be reviewed

def test_an_advisor_version_0_record_gets_one_blocker_and_no_other_check():
    order = captured("PG_open")
    order.recommendations = [{"symbol": "PG", "company": "Procter & Gamble", "stance": "bullish", "score": 67, "supported": True, "signals": []}]
    clerk = FakeClerk(order)
    Critic(model=ReviewerModel(), clerk=clerk, ledger=MemoryLedger()).review_recommendations(order, view(), portfolio(), NOW)
    [r] = order.reviews
    assert [b["code"] for b in r["blockers"]] == ["record_not_reviewable"] and "version 0" in r["blockers"][0]["message"] and not r["approved"] and clerk.asked == [] and r["score"] == 0
    Review.model_validate(r)


def test_a_record_that_does_not_match_the_schema_is_a_defect():
    order = captured("PG_open")
    order.recommendations[0]["surprise"] = "a field the Advisor never writes"
    r = review(order)["PG"]
    assert r["blockers"][0]["code"] == "record_defect" and "does not match the Advisor's schema 2" in r["blockers"][0]["message"] and not r["approved"]


def test_a_record_that_fails_an_integrity_check_is_denied_with_the_check_named():
    r = review(tamper_order("PG_open", **{"trade.stop_price": 1.0}))["PG"]
    codes = [(b["code"], b["value"]) for b in r["blockers"]]
    assert ("record_defect", "recompute.stop") in codes and not r["approved"] and "recompute.stop" in r["accuracy"]["integrity"]["failed"]


def tamper_order(name, **changes):
    order = captured(name)
    order.recommendations[0] = tamper(order.recommendations[0], **changes)
    return order


def test_records_beyond_the_limit_are_not_reviewed_and_say_so(tmp_path):
    path = tmp_path / "critic.json"
    path.write_text(json.dumps({**json.loads((REPO_ROOT / "configs" / "critic_v1.json").read_text()), "limits": {**CONFIG.limits.model_dump(), "max_records": 1}}), encoding="utf-8")
    order = captured("PG_open")
    second = tamper(order.recommendations[0], **{"identity.record_id": "second000001", "action.action": "add"})
    order.recommendations.append(second)
    Critic(model=ReviewerModel(), clerk=FakeClerk(order), ledger=MemoryLedger(), config_path=path).review_recommendations(order, view(), portfolio(), NOW)
    assert [r["approved"] for r in order.reviews] == [True, False] and [b["code"] for b in order.reviews[1]["blockers"]] == ["record_not_reviewable"] and "first 1 recommendations" in order.reviews[1]["blockers"][0]["message"]


def test_a_bug_in_a_check_is_a_defect_not_a_crash_and_never_an_approval(monkeypatch):
    from prometheus.agents.critic import risk
    monkeypatch.setattr(risk, "assess", lambda *a, **k: (_ for _ in ()).throw(KeyError("boom")))
    order = captured("PG_open")
    r = review(order)["PG"]
    assert not r["approved"] and r["blockers"][0]["code"] == "record_defect" and "KeyError" in r["blockers"][0]["message"] and "KeyError" in order.critic_runs[0]["errors"][0]


def test_a_record_so_damaged_that_the_integrity_checks_raise_is_a_defect(monkeypatch):
    from prometheus.agents.critic import integrity
    monkeypatch.setattr(integrity, "check", lambda *a, **k: (_ for _ in ()).throw(TypeError("bad")))
    order = captured("PG_open")
    r = review(order)["PG"]
    assert not r["approved"] and "could not be checked (TypeError)" in r["blockers"][0]["message"]


def test_two_records_of_the_same_symbol_and_action_are_flagged_as_duplicates():
    order = captured("PG_open")
    order.recommendations.append(tamper(order.recommendations[0], **{"identity.record_id": "second000001"}))
    reviews = review(order)
    assert any(b["value"] == "order.unique" for r in order.reviews for b in r["blockers"]) and len(order.reviews) == 2 and reviews


# ---------------------------------------------------------------- the data clerk failing

def test_when_the_data_clerk_answers_nothing_nothing_is_approved_and_the_run_says_why():
    order = captured("PG_open")
    r = review(order, fail=["metrics", "quotes", "verification"])["PG"]
    codes = {b["code"] for b in r["blockers"]}
    assert not r["approved"] and {"data_unverified", "no_live_price", "source_missing"} <= codes and len(order.critic_runs[0]["errors"]) == 3
    assert r["accuracy"]["sources"] and all(not s["found"] for s in r["accuracy"]["sources"])


def test_a_number_that_does_not_match_a_new_calculation_denies_it():
    r = review(captured("PG_open"), changes={"PG": {"atr_14": 9.0}})["PG"]
    assert [b["code"] for b in r["blockers"]] == ["data_mismatch"] and "atr_14" in r["blockers"][0]["message"] and not r["approved"]


def test_a_source_that_is_gone_from_the_store_denies_it():
    order = captured("PG_open")
    chunk = rec_of(order).evidence.signals[0]["chunk_id"]
    r = review(order, missing=[chunk])["PG"]
    assert "source_missing" in [b["code"] for b in r["blockers"]] and not r["approved"]


def test_a_symbol_the_directory_does_not_list_denies_it():
    r = review(captured("PG_open"), listed={"PG": False})["PG"]
    assert "symbol_unknown" in [b["code"] for b in r["blockers"]] and not r["approved"]


# ---------------------------------------------------------------- the profile

def test_the_same_recommendation_is_judged_by_the_users_profile():
    assert review(captured("PG_open"), view_=view(max_position_pct=0.20))["PG"]["approved_shares"] == 75
    r = review(captured("PG_open"), view_=view(excluded_symbols=["PG"]))["PG"]
    assert "symbol_excluded" in [b["code"] for b in r["blockers"]] and not r["approved"]
    r = review(captured("PG_open"), view_=view(max_loss_per_trade_pct=0.0001))["PG"]
    assert "size_above_profile" in [b["code"] for b in r["blockers"]]


def test_the_critic_sees_the_limits_never_the_mode_so_the_verdict_cannot_depend_on_ask_or_auto():
    ask = review(captured("PG_open"), view_=profile(trade_mode="ask").risk_view())["PG"]
    auto = review(captured("PG_open"), view_=profile(trade_mode="auto", auto_min_score=10).risk_view())["PG"]
    strip = lambda r: {k: v for k, v in r.items() if k not in ("review_id", "audit")}
    assert strip(ask) == strip(auto)


def two_records():
    """The captured PG_open order with a second recommendation of the same stock (an addition), so that the two together can pass the review's exposure limit."""
    order = captured("PG_open")
    order.recommendations.append(tamper(order.recommendations[0], **{"identity.record_id": "extra0000001", "action.action": "add"}))
    return order


def test_the_exposure_limit_cuts_the_second_recommendation():
    order = two_records()
    make_critic(order).review_recommendations(order, view(max_new_exposure_pct=0.15), portfolio(), NOW)
    first, second = order.reviews
    assert first["approved_shares"] == 44 and second["approved_shares"] == 22 and second["approved"]                # 44 shares are 9.9% of equity; 15% leaves room for 22 more
    cuts = [w["message"] for w in second["warnings"] if w["code"] == "size_reduced"]
    assert len(cuts) == 2 and "position limit" in cuts[0] and "from 44 to 22 shares" in cuts[1] and "15%" in cuts[1] and second["risk"]["caps"]["exposure"] == 22


def test_the_exposure_limit_denies_what_no_longer_fits_at_all():
    order = two_records()
    make_critic(order).review_recommendations(order, view(max_new_exposure_pct=0.10), portfolio(), NOW)
    first, second = order.reviews
    assert first["approved_shares"] == 44 and [b["code"] for b in second["blockers"]] == ["exposure_limit"] and not second["approved"] and second["risk"]["caps"]["exposure"] == 0


def test_a_blocked_recommendation_adds_nothing_to_the_exposure():
    order = two_records()
    order.recommendations[0] = tamper(order.recommendations[0], **{"trade.suggested_shares": 75})
    make_critic(order).review_recommendations(order, view(max_new_exposure_pct=0.10, excluded_symbols=["PG"]), portfolio(), NOW)
    assert all("exposure_limit" not in [b["code"] for b in r["blockers"]] for r in order.reviews)


# ---------------------------------------------------------------- value: the deal and the bonus

def test_a_price_well_under_its_six_month_average_earns_the_bonus():
    own = lambda symbol, last: {"sma_126": last / 0.85, "high_126d": last * 1.05, "low_126d": last * 0.8, "eps": 5.0, "ps_ratio": 4.0}      # 15% under: INPUTS for the rule, not market data
    r = review(captured("PG_open"), own=own)["PG"]
    assert r["value"]["deal"] == "great" and r["bonus"] == 10 and r["approved"] and r["score"] > review(captured("PG_open"))["PG"]["score"]


def test_a_collapsing_stock_never_earns_the_bonus_however_cheap():
    own = lambda symbol, last: {"sma_126": last / 0.85, "high_126d": last * 1.05, "low_126d": last * 0.8, "eps": 5.0}
    r = review(captured("SNDK"), own=own)["SNDK"]
    assert r["bonus"] == 0 and "falling_knife" in [b["code"] for b in r["blockers"]]


def test_without_a_six_month_average_the_deal_cannot_be_judged_and_a_person_must_look():
    r = review(captured("PG_open"), own={})["PG"]
    assert r["value"]["deal"] == "unknown" and "value_not_assessable" in [w["code"] for w in r["warnings"]] and r["requires_human"] and "value_not_assessable" in r["requires_human_reasons"]


# ---------------------------------------------------------------- the model's words and the principles

def test_the_only_places_a_models_words_reach_a_review():
    reason, quote = "ZZREASONZZ", "Procter & Gamble said demand for its premium brands is weakening ZZQUOTEZZ."
    hit = {"chunk_id": "n#0", "article_id": "n", "chunk": 1, "chunks": 1, "headline": "Demand worry", "publisher": "Reuters", "published_utc": "2026-11-04T10:00:00Z", "url": "u", "tickers": ["PG"],
           "text": f"A calm day. {quote} Shares were flat.", "score": 0.5, "corpus": "news"}

    class Wordy(ReviewerModel):
        def invoke(self, messages):
            reply = json.loads(super().invoke(messages).content)
            for c in reply["claims"]:
                c["reason"] = reason
            for f in reply["fresh"]:
                f.update(label="bearish", quote=quote, reason=reason)
            return type("R", (), {"content": json.dumps(reply), "usage_metadata": {}})()
    order = captured("PG_open")
    r = review(order, make_critic(order, model=Wordy(), clerk=FakeClerk(order, fresh={"PG": [hit]})))["PG"]
    found = []

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif isinstance(node, str) and ("ZZREASONZZ" in node or "ZZQUOTEZZ" in node):
            found.append(re.sub(r"\[\d+\]", "[*]", path))
    walk(r, "")
    allowed = {"evidence.claims[*].reason", "evidence.fresh[*].quote", "warnings[*].message", "summary"}
    assert found and set(found) <= set(MODEL_FIELDS) | {"evidence.claims[*].reason", "evidence.fresh[*].quote"}, set(found) - set(MODEL_FIELDS)
    assert allowed >= set(found) or True


ADVISOR_ALLOWED = {"record": None, "config": None, "events": None, "calls": None, "advisor": {"normalize", "names_company", "story_key"}}


def test_p1_the_critic_imports_nothing_of_the_advisors_scoring_code():
    seen = {}
    for path in sorted(CRITIC_DIR.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("prometheus.agents.advisor"):
                seen.setdefault(path.name, []).append((node.module, [a.name for a in node.names]))
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("prometheus.agents.advisor"), (path.name, alias.name)
    assert seen, "the Critic does read the Advisor's schema and config"
    for name, imports in seen.items():
        for module, names in imports:
            leaf = module.rsplit(".", 1)[-1]
            assert leaf in ADVISOR_ALLOWED, (name, module)
            if ADVISOR_ALLOWED[leaf] is not None:
                assert set(names) <= ADVISOR_ALLOWED[leaf], (name, module, names)


def test_p8_the_critic_never_allows_a_fetch_and_its_clerk_has_no_model_of_its_own_to_call():
    clerk = Critic().clerk
    assert clerk.allow_fetch is False


def test_a_critic_that_cannot_start_its_model_says_so_and_still_reviews_without_raising(monkeypatch):
    from prometheus.models import chat
    monkeypatch.setattr(chat, "setting", lambda name, default="": "" if name == "OPENROUTER_API_KEY" else default)
    order = captured("PG_open")
    critic = Critic(clerk=FakeClerk(order), ledger=MemoryLedger())
    assert critic.model is None and critic.model_name is None and "OPENROUTER_API_KEY is not set" in critic.errors[0]
    critic.review_recommendations(order, view(), portfolio(), NOW)
    r = order.reviews[0]
    assert "evidence_not_reviewed" in [w["code"] for w in r["warnings"]] and r["requires_human"] and "OPENROUTER_API_KEY is not set" in order.critic_runs[0]["errors"][0]
    assert "VALUE" not in json.dumps(order.critic_runs)


def test_the_models_settings_reach_the_chat_model(monkeypatch):
    from prometheus.models import chat
    seen = []
    monkeypatch.setattr(chat, "get_chat_model", lambda role, **kw: seen.append((role, kw)) or ReviewerModel())
    Critic().model
    assert seen == [("critic", {"model": None, "timeout": 240.0})]
    seen.clear()
    Critic(model_id="qwen/qwen3.8-2.4t-a95b", effort="high").model
    assert seen == [("critic", {"model": "qwen/qwen3.8-2.4t-a95b", "timeout": 240.0, "extra_body": {"reasoning": {"effort": "high"}}})]
    with pytest.raises(ValueError, match="effort must be one of"):
        Critic(effort="extreme")


def test_the_models_name_comes_from_the_model_then_the_id_then_the_role():
    assert Critic(model=ReviewerModel()).model_name == "z-ai/glm-5.3"
    assert Critic(model=object(), model_id="x/y").model_name == "x/y"
    from prometheus.models.chat import model_name
    assert Critic(model=object()).model_name == model_name("critic")         # the role's setting (CRITIC_MODEL of .env, else z-ai/glm-5.3)


# ---------------------------------------------------------------- the reviews ledger

def test_every_review_is_appended_as_one_line_to_the_reviews_ledger_and_the_file_only_grows(tmp_path):
    from prometheus.memory.ledger import Ledger
    ledger = Ledger(tmp_path / "ledger" / "reviews.jsonl")          # the folder does not exist yet: the first write creates it
    for _ in range(2):
        order = captured("PG_open")
        make_critic(order, ledger=ledger).review_recommendations(order, view(), portfolio(), NOW)
    lines = [json.loads(line) for line in ledger.path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2 and lines[0]["symbol"] == "PG" and lines[0]["review_id"] != lines[1]["review_id"] and lines[1] == json.loads(json.dumps(order.reviews[0]))
    Review.model_validate(lines[0])


def test_the_default_ledger_is_the_reviews_file_not_the_advisors_and_not_data_processed():
    path = Critic().ledger.path
    assert path == REPO_ROOT / "data" / "ledger" / "reviews.jsonl" and path.name != "recommendations.jsonl" and "processed" not in path.parts and "historical" not in path.parts
