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


def test_the_critics_blocker_codes_have_no_size_and_the_review_has_no_field_for_one():
    from prometheus.agents.critic.record import Review, RiskDetail
    codes = {code for codes in BLOCKER_CODES.values() for code in codes}
    assert not {"size_above_profile", "exposure_limit"} & codes
    assert not {"approved_shares", "suggested_shares"} & set(Review.model_fields) and not {"equity", "equity_basis", "caps", "allowed_shares"} & set(RiskDetail.model_fields)
    with pytest.raises(ValueError):
        RiskDetail(caps={"position": 8})                                                                         # `extra="forbid"`: a size cannot be put back by accident


@pytest.mark.parametrize("edit, message", [
    (lambda d: d["weights"].update(accuracy=31), "weights must add up to 100"),
    (lambda d: d["weights"].pop("value"), "weights needs exactly"),
    (lambda d: d["risk"]["measures"][0].update(weight=30), "risk.measures weights"),
    (lambda d: d["value"]["measures"][0].update(weight=1), "value.measures weights"),
    (lambda d: d["accuracy"]["parts"].update(metrics=0.9), "accuracy.parts"),
    (lambda d: d["evidence"]["parts"].update(support=0.9), "evidence.parts"),
    (lambda d: d["verdict"].update(approve_min_score=70), "approve_min_score"),                                    # Critic v3 (rule V8): the score no longer denies, so there is no mark; the old key is an unknown key
    (lambda d: d["verdict"]["grades"].update(B=95), "must fall from A to D"),
    (lambda d: d["verdict"]["grades"].pop("D"), "exactly A, B, C and D"),
    (lambda d: d["risk"]["levels"].update(medium=10), "low < medium < high"),
    (lambda d: d["risk"]["allowed"].update(moderate="very_high"), "very_high is allowed to nobody"),
    (lambda d: d["risk"]["age_caps"].append({"min_age": 80, "tolerance": "reckless"}), "tolerance that does not exist"),
    (lambda d: d["risk"]["measures"][0].update(safe=0.6, risky=0.6), "different safe and risky"),
        (lambda d: d["value"].update(rich_premium=0.5), "rich_premium must be under"),
    (lambda d: d["value"].update(good_discount=0.5), "good_discount must be under"),
    (lambda d: d["value"]["bonus"].update(zero_at=0.2), "zero_at must be under"),
    (lambda d: d["chart"]["patterns"].pop(), "chart.patterns needs exactly one entry"),
    (lambda d: d["chart"]["patterns"][0].update(points=-15), "golden_cross is bullish: its points must be positive"),
    (lambda d: next(p for p in d["chart"]["patterns"] if p["name"] == "obv_falling").update(points=5), "obv_falling is bearish: its points must be negative"),
    (lambda d: next(p for p in d["chart"]["patterns"] if p["name"] == "double_bottom").update(points=-10), "double_bottom is bullish: its points must be positive"),
    (lambda d: next(p for p in d["chart"]["patterns"] if p["name"] == "breakdown_52w").update(points=10), "breakdown_52w is bearish: its points must be negative"),
    (lambda d: d["chart"]["patterns"].remove(next(p for p in d["chart"]["patterns"] if p["name"] == "head_and_shoulders")), "chart.patterns needs exactly one entry"),
    (lambda d: d["weights"].pop("analysts"), "weights needs exactly"),
    (lambda d: d["analysts"].update(store="/etc/analysts"), "analysts.store must be a folder under data/"),
    (lambda d: d["analysts"].update(store="data/../secrets"), "analysts.store must be a folder under data/"),
    (lambda d: d["analysts"].update(bearish_drop=0.5), "bearish_drop"),
    (lambda d: d["analysts"].update(min_analysts=0), "min_analysts"),
    (lambda d: d["analysts"].pop("trend_months"), "trend_months"),
    (lambda d: d["performance"].pop("poor_ret_63d"), "poor_ret_63d"),
    (lambda d: d["chart"].update(bearish_at=5), "bearish_at"),
    (lambda d: d["chart"]["bonus"].pop("max_points"), "max_points"),
    (lambda d: d["chart"].update(level_tolerance=1.5), "level_tolerance"),
    (lambda d: d["chart"].pop("pivot_window"), "pivot_window"),
    (lambda d: d["chart"].pop("pattern_recent_sessions"), "pattern_recent_sessions"),
    (lambda d: d["chart"].update(pattern_recent_sessions=0), "pattern_recent_sessions"),
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


def test_the_closed_list_of_blocker_codes_has_every_code_once():
    assert len(BLOCKER_ORDER) == len(set(BLOCKER_ORDER)) == 26          # Critic v3: `data_unreliable` joined the accuracy's, `data_unverified` and `data_mismatch` left (rule V7); the analysts' dimension has no blocker
    assert set(BLOCKER_CODES) == set(cc.DIMENSIONS) and BLOCKER_ORDER[0] == "record_not_reviewable" and BLOCKER_ORDER[-1] == "fresh_negative_event" and BLOCKER_CODES["analysts"] == ()


def test_the_weights_of_version_3_are_the_specs_and_total_a_hundred():
    config = cc.load_config()
    assert config.config_version == "3" and dict(config.weights) == {"accuracy": 25, "evidence": 15, "risk_fit": 15, "timing": 10, "value": 15, "chart": 10, "analysts": 10} and sum(config.weights.values()) == 100
    assert not hasattr(config.verdict, "approve_min_score") and config.verdict.blocked_score_cap == 49


def test_the_new_numbers_of_version_3_are_the_specs():
    config = cc.load_config()
    a, c = config.analysts, config.chart
    assert (a.min_analysts, a.trend_months, a.bearish_mean, a.bearish_drop, a.store) == (3, 3, 2.5, -0.5, "data/analysts") and config.performance.poor_ret_63d == -0.15
    assert (c.bearish_at, c.bonus.per_point, c.bonus.max_points, c.pivot_window, c.pattern_lookback, c.pattern_recent_sessions, c.level_tolerance, c.min_gap, c.min_depth, c.year_sessions) == (-20, 0.25, 8, 5, 120, 10, 0.03, 10, 0.05, 252)
    assert {p.name: p.points for p in c.patterns if p.name in ("double_bottom", "double_top", "inverse_head_and_shoulders", "head_and_shoulders", "ascending_triangle", "descending_triangle", "breakout_52w", "breakdown_52w")} == {
        "double_bottom": 10, "double_top": -10, "inverse_head_and_shoulders": 10, "head_and_shoulders": -10, "ascending_triangle": 5, "descending_triangle": -5, "breakout_52w": 10, "breakdown_52w": -10}
