"""The Critic's settings file (configs/critic_v1.json): read once, checked completely (Critic spec Appendix A)."""
import json

import pytest

from prometheus.agents.critic import config as cc
from prometheus.agents.critic.record import BLOCKER_CODES, BLOCKER_ORDER
from prometheus.rag.config import REPO_ROOT

FILE = REPO_ROOT / "configs" / "critic_v1.json"


def raw() -> dict:
    return json.loads(FILE.read_text(encoding="utf-8"))


def write(tmp_path, data) -> str:
    path = tmp_path / "critic.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_the_shipped_file_loads_and_holds_the_specs_numbers():
    c = cc.load_config()
    assert c.weights == {"accuracy": 30, "evidence": 15, "risk_fit": 20, "timing": 10, "value": 15, "chart": 10}
    assert (c.verdict.approve_min_score, c.verdict.blocked_score_cap, c.verdict.auto_min_score, c.verdict.unknown_credit) == (70, 49, 80, 0.5)
    assert c.risk.levels == {"low": 20, "medium": 45, "high": 70} and c.risk.allowed == {"conservative": "low", "moderate": "medium", "aggressive": "high"}
    assert (c.value.max_premium, c.value.bonus.max_points) == (0.30, 10) and [m.name for m in c.value.measures] == ["discount_to_avg_6m", "reward_to_risk", "earnings_yield"]
    assert (c.chart.neutral, c.chart.weak_below, c.chart.dead_cat.min_decline, c.chart.dead_cat.max_trough_age) == (50, 35, -0.25, 15) and len(c.chart.patterns) == 15
    assert c.limits.model_timeout_seconds == 240 and len(c.evidence.control_texts) == 3 and c.ledger.path == "data/ledger/reviews.jsonl"


def test_the_weights_of_each_score_add_up_to_what_they_must():
    c = cc.load_config()
    assert sum(c.weights.values()) == 100 and sum(m.weight for m in c.risk.measures) == 100 and sum(m.weight for m in c.value.measures) == 100
    assert c.accuracy.parts.metrics + c.accuracy.parts.sources + c.accuracy.parts.cross == pytest.approx(1.0)
    assert c.evidence.parts.support + c.evidence.parts.clean + c.evidence.parts.breadth == pytest.approx(1.0)


def test_a_blocked_score_can_never_reach_the_approval_mark():
    assert cc.load_config().verdict.blocked_score_cap < cc.load_config().verdict.approve_min_score


@pytest.mark.parametrize("edit, message", [
    (lambda d: d["weights"].update(accuracy=31), "weights must add up to 100"),
    (lambda d: d["weights"].pop("value"), "weights needs exactly"),
    (lambda d: d["risk"]["measures"][0].update(weight=30), "risk.measures weights"),
    (lambda d: d["value"]["measures"][0].update(weight=1), "value.measures weights"),
    (lambda d: d["accuracy"]["parts"].update(metrics=0.9), "accuracy.parts"),
    (lambda d: d["evidence"]["parts"].update(support=0.9), "evidence.parts"),
    (lambda d: d["verdict"].update(blocked_score_cap=70), "blocked_score_cap must be under approve_min_score"),
    (lambda d: d["verdict"]["grades"].update(B=95), "must fall from A to D"),
    (lambda d: d["verdict"]["grades"].pop("D"), "exactly A, B, C and D"),
    (lambda d: d["risk"]["levels"].update(medium=10), "low < medium < high"),
    (lambda d: d["risk"]["allowed"].update(moderate="very_high"), "very_high is allowed to nobody"),
    (lambda d: d["risk"]["age_caps"].append({"min_age": 80, "tolerance": "reckless"}), "tolerance that does not exist"),
    (lambda d: d["risk"]["measures"][0].update(safe=0.6, risky=0.6), "different safe and risky"),
    (lambda d: d["risk"]["measures"][6].update(safe=-1), "logarithmic"),
    (lambda d: d["value"].update(rich_premium=0.5), "rich_premium must be under"),
    (lambda d: d["value"].update(good_discount=0.5), "good_discount must be under"),
    (lambda d: d["value"]["bonus"].update(zero_at=0.2), "zero_at must be under"),
    (lambda d: d["chart"]["patterns"].pop(), "chart.patterns needs exactly one entry"),
    (lambda d: d["chart"]["patterns"][0].update(points=-15), "golden_cross is bullish: its points must be positive"),
    (lambda d: d["chart"]["patterns"][-1].update(points=5), "obv_falling is bearish: its points must be negative"),
    (lambda d: d["chart"]["patterns"][0].update(name="made_up"), "chart.patterns needs exactly one entry"),
    (lambda d: d["chart"].update(rsi_healthy_high=85), "rsi_healthy_low < rsi_healthy_high < rsi_overbought"),
    (lambda d: d["chart"]["dead_cat"].update(min_decline=0.25), "min_decline"),
    (lambda d: d["chart"].pop("neutral"), "neutral"),
    (lambda d: d["evidence"]["label_weight"].pop("weak"), "label_weight needs exactly"),
    (lambda d: d["evidence"]["label_weight"].update(weak=1.5), "label_weight values"),
    (lambda d: d["evidence"].update(control_texts=[]), "control_texts"),
    (lambda d: d["timing"]["session_factor"].update(closed=1.0), "session_factor needs exactly"),
    (lambda d: d["accuracy"]["tolerance"].pop("bars_changed"), "accuracy.tolerance needs"),
    (lambda d: d.update(colour="red"), "colour"),
    (lambda d: d["limits"].update(max_recrods=3), "max_recrods"),
    (lambda d: d["limits"].pop("max_records"), "max_records"),
])
def test_a_bad_file_is_refused_when_it_is_loaded(tmp_path, edit, message):
    data = raw()
    edit(data)
    with pytest.raises(ValueError, match=message):
        cc.load_config(write(tmp_path, data))


def test_a_missing_or_broken_file_is_named(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        cc.load_config(tmp_path / "nope.json")
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        cc.load_config(tmp_path / "bad.json")


def test_the_note_key_is_ignored_and_the_config_cannot_be_changed():
    c = cc.load_config()
    with pytest.raises(Exception):
        c.verdict.approve_min_score = 0


def test_the_closed_list_of_blocker_codes_has_every_code_once():
    assert len(BLOCKER_ORDER) == len(set(BLOCKER_ORDER)) == 29
    assert set(BLOCKER_CODES) == set(cc.DIMENSIONS) and BLOCKER_ORDER[0] == "record_not_reviewable" and BLOCKER_ORDER[-1] == "fresh_negative_event"
