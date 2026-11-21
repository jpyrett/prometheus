"""The whole Critic step with fakes (Critic spec section 7): the nine steps over the real captures, and the principles P1 to P8."""
import ast
import copy
import json
from datetime import datetime

import pytest
from advisor_helpers import MemoryLedger
from critic_helpers import CONFIG, NOW, FakeClerk, ReviewerModel, at, captured, make_critic, portfolio, profile, review, tamper, view

from prometheus.agents.critic.critic import Critic
from prometheus.agents.critic.record import Review
from prometheus.rag.config import REPO_ROOT

CRITIC_DIR = REPO_ROOT / "prometheus" / "agents" / "critic"


# ---------------------------------------------------------------- the step


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


# ---------------------------------------------------------------- the data clerk failing

def test_when_the_data_clerk_answers_nothing_nothing_is_approved_and_the_run_says_why():
    order = captured("PG_open")
    r = review(order, fail=["metrics", "quotes", "verification"])["PG"]
    codes = {b["code"] for b in r["blockers"]}
    assert not r["approved"] and {"data_unreliable", "no_live_price", "source_missing"} <= codes and len(order.critic_runs[0]["errors"]) == 3      # Critic v3 (V7): no numbers at all is `data_unreliable` (a fault), it was `data_unverified`
    assert r["accuracy"]["sources"] and all(not s["found"] for s in r["accuracy"]["sources"])


def test_a_number_that_does_not_match_a_new_calculation_is_a_warning_not_a_denial_but_the_price_denies():
    """AMENDED (Critic v3, rule V7: no nitpicking on one financial metric): a decisive number (the ATR) that differs from a new calculation was a `data_mismatch` BLOCKER and denied; it is a WARNING now (the accuracy score still counts
    it). The PRICE itself (`last_close`) not matching is the record's fault: `data_unreliable`, a blocker that denies alone."""
    r = review(captured("PG_open"), changes={"PG": {"atr_14": 9.0}})["PG"]
    assert r["approved"] and r["blockers"] == [] and r["denial_reasons"] == [] and r["accuracy"]["match_rate"] < 1.0
    [warn] = [w for w in r["warnings"] if w["code"] == "data_mismatch"]
    assert "atr_14" in warn["message"] and "data_unreliable" not in [w["code"] for w in r["warnings"]]
    assert r["dimensions"]["accuracy"]["score"] < 100
    price = review(captured("PG_open"), changes={"PG": {"last_close": 150.0}})["PG"]
    assert not price["approved"] and "data_unreliable" in price["denial_reasons"] and next(b for b in price["blockers"] if b["code"] == "data_unreliable")["message"].startswith("The price the recommendation was scored from does not match")


def test_a_symbol_the_directory_does_not_list_denies_it():
    r = review(captured("PG_open"), listed={"PG": False})["PG"]
    assert "symbol_unknown" in [b["code"] for b in r["blockers"]] and not r["approved"]


# ---------------------------------------------------------------- the profile


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


# ---------------------------------------------------------------- the model's words and the principles


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


def test_the_default_ledger_is_the_reviews_file_not_the_advisors_and_not_data_processed():
    path = Critic().ledger.path
    assert path == REPO_ROOT / "data" / "ledger" / "reviews.jsonl" and path.name != "recommendations.jsonl" and "processed" not in path.parts and "historical" not in path.parts


# ---------------------------------------------------------------- the Critic judges recommendations to buy and nothing else (entry 105)


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
