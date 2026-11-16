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
    returned = make_critic(order, ledger=ledger).review_recommendations(order, view(), NOW)
    assert returned is order and order.status == "reviewed" and order.schema_version == 3
    assert [r["record_id"] for r in order.reviews] == [r["identity"]["record_id"] for r in order.recommendations] and len(order.reviews) == 1
    assert dump(order) == before                                         # P5: the Advisor's own fields are never touched, byte for byte (the only addition is the `critic` block)
    review_ = Review.model_validate(order.reviews[0])
    assert review_.approved and review_.symbol == "PG" and review_.action == "open" and review_.request_id == order.request_id and review_.reviewed_utc == "2026-11-05T15:30:00Z"
    assert ledger.records == order.reviews and ledger.problems == []      # one line per review, in the reviews ledger


def test_the_run_is_recorded_on_the_order():
    order = captured("PG_open")
    model = ReviewerModel()
    make_critic(order, model=model).review_recommendations(order, view(), NOW)
    [run] = order.critic_runs
    assert (run["stage"], run["model"], run["records"], run["calls"], run["records_called"], run["controls_failed"], run["error"]) == ("review", "z-ai/glm-5.3", 1, 1, 1, 0, None)
    assert run["tokens_in"] == 400 and run["tokens_out"] == 80 and run["independent_model"] is True and run["advisor_model"] and run["tolerance"] == "moderate"
    assert run["now_et"] == "2026-11-05 10:30 ET" and run["session"] == "regular" and run["seconds"] >= 0 and run["metric_order"] and run["quote_order"] and run["verify_order"]
    assert order.reviews[0]["audit"]["critic_model"] == "z-ai/glm-5.3" and order.reviews[0]["audit"]["config_version"] == CONFIG.config_version and order.reviews[0]["audit"]["verify_order"] == run["verify_order"]


def test_a_healthy_recommendation_is_approved_for_an_hour_and_the_review_carries_no_size():
    """Design choices entry 80: the Critic approves a recommendation, not a number of shares. The risk-fit points are 100 x (1 - 0.5 x 5.2 / 45) = 94.22 of 20 = 18.84 (PG's risk score is 5.2 by hand: tests/test_critic_risk.py)."""
    r = review(captured("PG_open"))["PG"]
    assert r["approved"] and r["blockers"] == [] and r["valid_until_utc"] == "2026-11-05T16:30:00Z"
    assert not {"approved_shares", "suggested_shares"} & set(r) and not {"equity", "equity_basis", "caps", "allowed_shares"} & set(r["risk"]) and "size_reduced" not in [w["code"] for w in r["warnings"]]
    assert (r["score"], r["grade"]) == (88, "B") and r["summary"].startswith("APPROVE PG: score 88 (B)") and "share" not in r["summary"]
    assert {n: d["points"] for n, d in r["dimensions"].items()} == {"accuracy": 30.0, "evidence": 15.0, "risk_fit": 18.56, "timing": 10.0, "value": 6.58, "chart": 7.5}


def test_every_check_runs_for_every_record_even_when_the_first_blocker_is_found():
    r = review(captured("PG_open"), now=at("2026-11-07 12:00"))["PG"]                 # a Saturday
    assert not r["approved"] and {"market_closed", "record_expired"} <= {b["code"] for b in r["blockers"]} and r["valid_until_utc"] is None and "approved_shares" not in r
    assert set(r["dimensions"]) == {"accuracy", "evidence", "risk_fit", "timing", "value", "chart"} and r["dimensions"]["accuracy"]["score"] == 100 and r["dimensions"]["risk_fit"]["score"] > 0
    assert r["score"] == 49 and r["score_before_blockers"] > 49 and r["grade"] == "F" and r["timing"]["session"] == "closed"
    assert "market_closed" in r["denial_reasons"] and r["summary"].startswith("DENY PG: ")


def test_ignore_market_hours_drops_the_closed_market_blocker_from_the_review(monkeypatch):
    monkeypatch.setenv("IGNORE_MARKET_HOURS", "true")
    order = captured("PG_open")
    make_critic(order).review_recommendations(order, view(), at("2026-11-07 12:00"))
    r = order.reviews[0]
    assert "market_closed" not in {b["code"] for b in r["blockers"]} and any(n["code"] == "market_hours_ignored" for n in r["notes"])
    assert "record_expired" in {b["code"] for b in r["blockers"]} and order.critic_runs[-1]["ignore_market_hours"] is True
    from prometheus.agents.critic.report import header
    assert "IGNORE_MARKET_HOURS" in header(order)


def test_a_review_twice_is_the_same_review():
    order = captured("PG_open")
    ledger = MemoryLedger()
    critic = make_critic(order, ledger=ledger)
    critic.review_recommendations(order, view(), NOW)
    first = copy.deepcopy(order.reviews)
    critic.review_recommendations(order, view(), NOW)
    assert order.reviews == first and len(order.critic_runs) == 1 and len(ledger.records) == 1
    order.status = "decided"
    assert critic.review_recommendations(order, view(), NOW).reviews == first


def test_a_traded_order_comes_back_from_the_critic_unchanged_and_without_a_message():
    """Trader v1 spec section 11 (phase 3): `traded` follows `decided`; the Critic hands such an order back as it does a `reviewed` or a `decided` one, not with "needs an order with recommendations"."""
    order = captured("PG_open")
    critic = make_critic(order)
    critic.review_recommendations(order, view(), NOW)
    order.status = "traded"
    before, errors, asked = copy.deepcopy(order.model_dump()), list(order.errors), list(critic.clerk.asked)
    out = critic.review_recommendations(order, view(), NOW)
    assert out is order and out.status == "traded" and out.errors == errors == [] and out.model_dump() == before and critic.clerk.asked == asked


def test_an_order_that_is_not_recommended_is_refused_with_a_message_and_nothing_else_happens():
    order = captured("PG_open")
    order.status = "data_ready"
    clerk = FakeClerk(order)
    Critic(model=ReviewerModel(), clerk=clerk, ledger=MemoryLedger()).review_recommendations(order, view(), NOW)
    assert order.status == "data_ready" and order.reviews == [] and "the Critic needs an order with recommendations (status data_ready)" in order.errors and clerk.asked == []


def test_no_recommendations_is_a_result_not_a_failure():
    order = captured("PG_open")
    order.recommendations = []
    clerk = FakeClerk(order)
    Critic(model=ReviewerModel(), clerk=clerk, ledger=MemoryLedger()).review_recommendations(order, view(), NOW)
    assert order.status == "reviewed" and order.reviews == [] and clerk.asked == [] and order.errors == [] and order.critic_runs[0]["records"] == 0


def test_a_clock_without_a_timezone_is_a_programming_mistake():
    order = captured("PG_open")
    with pytest.raises(ValueError, match="timezone"):
        make_critic(order).review_recommendations(order, view(), datetime(2026, 11, 5, 10, 30))
    assert order.status == "recommended"


# ---------------------------------------------------------------- records that cannot be reviewed

def test_an_advisor_version_0_record_gets_one_blocker_and_no_other_check():
    order = captured("PG_open")
    order.recommendations = [{"symbol": "PG", "company": "Procter & Gamble", "stance": "bullish", "score": 67, "supported": True, "signals": []}]
    clerk = FakeClerk(order)
    Critic(model=ReviewerModel(), clerk=clerk, ledger=MemoryLedger()).review_recommendations(order, view(), NOW)
    [r] = order.reviews
    assert [b["code"] for b in r["blockers"]] == ["record_not_reviewable"] and "version 0" in r["blockers"][0]["message"] and not r["approved"] and clerk.asked == [] and r["score"] == 0
    Review.model_validate(r)


def test_a_record_that_does_not_match_the_schema_is_a_defect():
    order = captured("PG_open")
    order.recommendations[0]["surprise"] = "a field the Advisor never writes"
    r = review(order)["PG"]
    assert r["blockers"][0]["code"] == "record_defect" and "does not match the Advisor's record schema" in r["blockers"][0]["message"] and not r["approved"]      # AMENDED (entry 105): the record schema is version 3, and the message no longer names a number


def test_an_old_record_that_still_carries_the_advisors_size_is_a_defect_because_the_record_has_no_size_now():
    """Entry 80: the Advisor's record has no `suggested_shares` or `position_pct` any more (the schema version stays 2, as when Amendment 7 took a field out). A work order saved before that and reviewed again is
    refused by the schema, and says which field: the models forbid unknown fields."""
    order = tamper_order("PG_open", **{"trade.suggested_shares": 7, "trade.position_pct": 0.1})
    r = review(order)["PG"]
    assert r["blockers"][0]["code"] == "record_defect" and "does not match the Advisor's record schema" in r["blockers"][0]["message"] and "suggested_shares" in r["blockers"][0]["message"] and not r["approved"]


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
    second = tamper(order.recommendations[0], **{"identity.record_id": "second000001"})
    order.recommendations.append(second)
    Critic(model=ReviewerModel(), clerk=FakeClerk(order), ledger=MemoryLedger(), config_path=path).review_recommendations(order, view(), NOW)
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
    r = review(captured("PG_open"), view_=view(excluded_symbols=["PG"]))["PG"]
    assert "symbol_excluded" in [b["code"] for b in r["blockers"]] and not r["approved"]
    r = review(captured("PG_open"), view_=view(risk_tolerance="conservative", age=72))["PG"]
    assert r["risk"]["tolerance_effective"] == "conservative" and r["risk"]["allowed_level"] == "low"


def test_the_critic_never_sees_the_mode_so_the_verdict_cannot_depend_on_ask_or_auto():
    ask = review(captured("PG_open"), view_=profile(trade_mode="ask").risk_view())["PG"]
    auto = review(captured("PG_open"), view_=profile(trade_mode="auto").risk_view())["PG"]
    strip = lambda r: {k: v for k, v in r.items() if k not in ("review_id", "audit")}
    assert strip(ask) == strip(auto)


def two_records():
    """The captured PG_open order with a second recommendation of the same stock (an addition), so that the two together can pass the review's exposure limit."""
    order = captured("PG_open")
    order.recommendations.append(tamper(order.recommendations[0], **{"identity.record_id": "extra0000001"}))
    return order


def test_each_recommendation_is_reviewed_on_its_own_there_is_no_running_exposure_across_the_review():
    """Design choices entry 80: the Critic's old step 9 (a running exposure across the review: `exposure_limit`, `size_reduced`) is gone with the size. Two records of one stock are two verdicts.
    AMENDED (entry 105): the second record was an `add` of the same stock and both were approved. There is no `add` now, and two records of one stock are a defect of the order (`order.unique`), so both are denied
    for that and for nothing else: still two reviews, each of its own, and no exposure across them."""
    order = two_records()
    make_critic(order).review_recommendations(order, view(), NOW)
    first, second = order.reviews
    assert [b["value"] for b in first["blockers"]] == [b["value"] for b in second["blockers"]] == ["order.unique"] and not first["approved"] and not second["approved"]
    assert not any(w["code"] == "size_reduced" for r in order.reviews for w in r["warnings"]) and not any("exposure" in b["code"] for r in order.reviews for b in r["blockers"])


# ---------------------------------------------------------------- value: the deal and the bonus

def test_a_price_well_under_its_six_month_average_earns_the_bonus():
    own = lambda symbol, last: {"sma_126": last / 0.85, "high_126d": last * 1.05, "low_126d": last * 0.8, "eps": 5.0, "ps_ratio": 4.0}      # 15% under: INPUTS for the rule, not market data
    r = review(captured("PG_open"), own=own)["PG"]
    assert r["value"]["deal"] == "great" and r["bonus"] == 10 and r["approved"] and r["score"] > review(captured("PG_open"))["PG"]["score"]


def test_a_collapsing_stock_never_earns_the_bonus_however_cheap():
    own = lambda symbol, last: {"sma_126": last / 0.85, "high_126d": last * 1.05, "low_126d": last * 0.8, "eps": 5.0}
    r = review(captured("SNDK"), own=own)["SNDK"]
    assert r["bonus"] == 0 and "falling_knife" in [b["code"] for b in r["blockers"]]


def test_without_a_six_month_average_the_deal_cannot_be_judged_and_that_is_a_note_not_a_denial():
    """AMENDED (Part F, entry 103 corrected: Jacob, 2026-11-15, a company with fewer than 126 bars must not be denied): the finding stays as a plain warning and asks for nobody."""
    r = review(captured("PG_open"), own={})["PG"]
    assert r["value"]["deal"] == "unknown" and "value_not_assessable" in [w["code"] for w in r["warnings"]] and not r["requires_human"] and r["requires_human_reasons"] == [] and r["approved"]


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


def test_the_critic_is_never_given_the_portfolio_or_the_users_money():
    """Design choices entry 80: the Advisor and the Critic only deal with recommendations. `review_recommendations(order, risk, now)` has no portfolio argument (a fourth argument is a TypeError), `review_order` and the
    Data Clerk requests of `recheck.gather` have none either, and nothing in the Critic's package imports the portfolio, the Account Manager or the account book."""
    import inspect

    from prometheus.agents.critic import recheck
    from prometheus.agents.critic.review import review_order
    assert list(inspect.signature(Critic.review_recommendations).parameters) == ["self", "order", "risk", "now"]
    assert list(inspect.signature(review_order).parameters) == ["critic", "order", "view", "now"]
    assert list(inspect.signature(recheck.gather).parameters) == ["clerk", "order", "items", "now", "config"]
    order = captured("PG_open")
    with pytest.raises(TypeError):
        make_critic(order).review_recommendations(order, view(), portfolio(), NOW)
    assert order.status == "recommended" and order.reviews == []                  # nothing was reviewed: the call was refused before it started
    for path in sorted(CRITIC_DIR.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [node.module or ""] + [f"{node.module}.{a.name}" for a in node.names] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names] if isinstance(node, ast.Import) else []
            assert not [n for n in names if n.startswith(("prometheus.common.portfolio", "prometheus.agents.account_manager", "prometheus.persistence"))], (path.name, names)


def test_the_critics_requests_to_the_data_clerk_name_only_the_recommended_symbols():
    """With no portfolio the Critic has no holdings to price: its quote order holds the recommended symbols and nothing else (the Account Manager asks for the holdings' prices itself)."""
    order = captured("PG_open")
    clerk = FakeClerk(order)
    make_critic(order, clerk=clerk).review_recommendations(order, view(), NOW)
    assert [a[1] for a in clerk.asked if a[0] == "quotes"] == [["PG"]] and all(a[1] == ["PG"] for a in clerk.asked if a[0] == "metrics")


def test_p8_the_critic_never_allows_a_fetch_and_its_clerk_has_no_model_of_its_own_to_call():
    clerk = Critic().clerk
    assert clerk.allow_fetch is False


def test_a_critic_that_cannot_start_its_model_says_so_and_still_reviews_without_raising(monkeypatch):
    from prometheus.models import chat
    monkeypatch.setattr(chat, "setting", lambda name, default="": "" if name == "OPENROUTER_API_KEY" else default)
    order = captured("PG_open")
    critic = Critic(clerk=FakeClerk(order), ledger=MemoryLedger())
    assert critic.model is None and critic.model_name is None and "OPENROUTER_API_KEY is not set" in critic.errors[0]
    critic.review_recommendations(order, view(), NOW)
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
        make_critic(order, ledger=ledger).review_recommendations(order, view(), NOW)
    lines = [json.loads(line) for line in ledger.path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2 and lines[0]["symbol"] == "PG" and lines[0]["review_id"] != lines[1]["review_id"] and lines[1] == json.loads(json.dumps(order.reviews[0]))
    Review.model_validate(lines[0])


def test_the_default_ledger_is_the_reviews_file_not_the_advisors_and_not_data_processed():
    path = Critic().ledger.path
    assert path == REPO_ROOT / "data" / "ledger" / "reviews.jsonl" and path.name != "recommendations.jsonl" and "processed" not in path.parts and "historical" not in path.parts


# ---------------------------------------------------------------- the Critic judges recommendations to buy and nothing else (entry 105)

def test_the_critic_refuses_to_review_a_record_that_is_not_a_purchase_and_asks_the_data_clerk_nothing():
    """Jacob (2026-11-15): *"The Critic never will approve or deny (judge) such a decision that affects the User's holdings."* The real PG_open record made a `reduce`, an `add` and a `hold` is not reviewed: it is a
    `record_defect` ("the Critic does not judge a holding or a sale"), denied, with no check made and no request to the Data Clerk."""
    for action in ("reduce", "add", "hold", "avoid"):
        order = tamper_order("PG_open", **{"action.action": action})
        clerk = FakeClerk(order)
        Critic(model=ReviewerModel(), clerk=clerk, ledger=MemoryLedger()).review_recommendations(order, view(), NOW)
        [r] = order.reviews
        assert not r["approved"] and r["action"] == action and [b["code"] for b in r["blockers"]] == ["record_defect"] and r["score"] == 0 and clerk.asked == []
        assert f"The record is not a recommendation to buy (its action is '{action}'): the Critic does not judge a holding or a sale." == r["blockers"][0]["message"]
        Review.model_validate(r)


def test_a_record_that_says_something_is_held_is_refused_with_the_reason_and_a_version_2_record_is_read():
    """A record that carries the user's holding (`held_shares`, `cost_basis`, `unrealized_pct`) is a defect: the record models refuse it. The same real record with the retired keys at their not-held values (what a version 2
    record has: `held_shares` 0, null cost basis and gain, `portfolio_fit` with its two numbers null and its note) is read and reviewed as before."""
    held = review(tamper_order("PG_open", **{"action.held_shares": 12, "action.cost_basis": 150.0}))["PG"]
    assert not held["approved"] and held["blockers"][0]["code"] == "record_defect" and "the record says something is held (held_shares 12)" in held["blockers"][0]["message"]
    order = captured("PG_open")
    record = tamper(order.recommendations[0], **{"action.held_shares": 0, "action.cost_basis": None, "action.unrealized_pct": None})
    record["portfolio_fit"] = {"sector_weight_after": None, "correlation_to_portfolio_63d": None, "note": "two names are one bet"}
    record.pop("concentration")
    order.recommendations[0] = record
    old = review(order)["PG"]
    assert old["approved"] and old["blockers"] == []
    for bad in ({"sector_weight_after": 0.3, "correlation_to_portfolio_63d": None, "note": None}, "a note"):
        order = captured("PG_open")
        order.recommendations[0] = {**{k: v for k, v in order.recommendations[0].items() if k != "concentration"}, "portfolio_fit": bad}
        r = review(order)["PG"]
        assert not r["approved"] and "the record carries portfolio information" in r["blockers"][0]["message"]


def test_no_module_of_the_critic_reads_what_the_user_holds_or_paid():
    """The Critic's rule that it is never given the portfolio (below) now covers the record too: no Critic module names `held_shares`, `cost_basis`, `unrealized_pct`, `held` or `portfolio_fit`, and none switches on an
    action other than `open`."""
    names = {"held_shares", "cost_basis", "unrealized_pct", "held", "portfolio_fit", "holding_action", "reduce", "adding_above_cost", "add_above_cost"}
    found = []
    for path in sorted((REPO_ROOT / "prometheus" / "agents" / "critic").glob("*.py")) + [REPO_ROOT / "prometheus" / "prompts" / "critic.py"]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docs = {id(n.body[0].value) for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef)) and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
        for node in ast.walk(tree):
            word = (node.id if isinstance(node, ast.Name) and node.id != "held" else node.attr if isinstance(node, ast.Attribute) else node.arg if isinstance(node, ast.arg) and node.arg != "held"
                    else node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs else None)      # a local variable called `held` is chart.py's "the pattern held", not a holding
            if word in names:
                found.append((path.name, word))
    assert found == []
    assert "add_above_cost" not in json.loads((REPO_ROOT / "configs" / "critic_v1.json").read_text())["value"]
