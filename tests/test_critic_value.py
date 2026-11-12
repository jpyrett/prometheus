"""Value: is the price a good deal? (Critic spec 8.5 and Appendix B.3.) The captures hold no six-month average, so the worked examples use invented INPUTS (said so in the names): the rules are what is tested."""
import pytest
from advisor_helpers import MemoryLedger
from critic_helpers import CONFIG, NOW, captured, rec_of, tamper

from prometheus.agents.critic import scoring as sc
from prometheus.agents.critic import value as vl

V, CREDIT = CONFIG.value, CONFIG.verdict.unknown_credit


def metrics(avg=100.0, high=105.0, ey=0.05, eps=5.0, ps=4.0, ret=0.01, dd=-0.05, low=90.0):
    def m(x):
        return {"value": x}
    return {"sma_126": m(avg), "high_126d": m(high), "low_126d": m(low), "earnings_yield": m(ey), "eps": m(eps), "ps_ratio": m(ps), "ret_21d": m(ret), "max_drawdown_63d": m(dd)}


def rec(stop=5.0, **changes):
    order = captured("PG_open")
    order.recommendations[0] = tamper(order.recommendations[0], **{"trade.stop_distance": stop, **changes})
    return rec_of(order)


def run(price=88.0, fetched=None, r=None, **kw):
    return vl.assess(r or rec(), fetched if fetched is not None else metrics(**kw), price, V, CREDIT)


def codes(result, severity=None):
    return [f.code for f in result.findings if severity in (None, f.severity)]


# ---------------------------------------------------------------- Appendix B.3

def test_a_discount_to_the_six_month_average_scores_96_and_is_great():
    """Amendment 3 removed the expected excess return from the value score: three measures, 50 / 30 / 20 (it was 40 / 25 / 20 / 15, a score of 79 with the excess)."""
    r = run(price=88.0, high=105.0, ey=0.05)
    by = {m.name: round(m.value * m.weight, 2) for m in r.measures}
    assert by == {"discount_to_avg_6m": 50.0, "reward_to_risk": 30.0, "earnings_yield": 15.79}
    assert r.score == pytest.approx(95.79, abs=0.01) and r.detail.deal == "great" and r.detail.discount == pytest.approx(0.12) and r.detail.reward_to_risk == pytest.approx(3.4)
    assert vl.bonus_points(r.discount, V.bonus) == 7.5 and codes(r) == []


def test_a_premium_scores_42_and_is_fair():
    r = run(price=108.0, high=112.0, ey=0.04, r=rec(stop=6.0))
    by = {m.name: round(m.value * m.weight, 2) for m in r.measures}
    assert by == {"discount_to_avg_6m": 27.5, "reward_to_risk": 2.5, "earnings_yield": 11.58}
    assert r.score == pytest.approx(41.58, abs=0.01) and r.detail.deal == "fair" and vl.bonus_points(r.discount, V.bonus) == 0


# ---------------------------------------------------------------- labels at their bounds

@pytest.mark.parametrize("premium, falling, label", [
    (-0.20, False, "great"), (-0.10, False, "great"), (-0.0999, False, "good"), (-0.03, False, "good"), (-0.0299, False, "fair"), (0.0, False, "fair"), (0.10, False, "fair"),
    (0.1001, False, "rich"), (0.30, False, "rich"), (0.3001, False, "overpriced"), (-0.5, True, "falling"), (0.5, True, "falling"), (None, False, "unknown"), (None, True, "falling")])
def test_the_label_is_the_first_that_fits(premium, falling, label):
    assert vl.label_of(premium, falling, V) == label


@pytest.mark.parametrize("discount, bonus", [(None, 0.0), (0.0, 0.0), (0.03, 0.0), (0.09, 5.0), (0.15, 10.0), (0.40, 10.0), (-0.10, 0.0)])
def test_the_bonus_grows_from_3_to_15_percent_under_the_average(discount, bonus):
    assert vl.bonus_points(discount, V.bonus) == bonus


# ---------------------------------------------------------------- the blockers

def test_overpriced_is_a_blocker_at_30_percent_above_the_average():
    assert codes(run(price=130.0), "blocker") == []                      # exactly 30%
    r = run(price=130.1)
    assert codes(r, "blocker") == ["overpriced"] and r.detail.deal == "overpriced" and "30%" in r.findings[0].message


def test_an_extreme_valuation_is_a_blocker_by_pe_and_by_sales_when_there_is_no_profit():
    assert "valuation_extreme" in codes(run(ey=0.012), "blocker") and "valuation_extreme" not in codes(run(ey=0.0125))
    r = run(eps=-1.0, ps=25.0)
    assert "valuation_extreme" in codes(r, "blocker") and "no profit" in [f for f in r.findings if f.code == "valuation_extreme"][0].message
    assert "valuation_extreme" not in codes(run(eps=-1.0, ps=20.0)) and "valuation_extreme" not in codes(run(eps=-1.0, ps=None))


@pytest.mark.parametrize("ret, dd, knife", [(-0.25, -0.05, True), (-0.2499, -0.05, False), (0.0, -0.45, True), (0.0, -0.4499, False), (-0.30, -0.50, True)])
def test_a_collapse_is_not_a_discount(ret, dd, knife):
    r = run(ret=ret, dd=dd)
    assert ("falling_knife" in codes(r, "blocker")) is knife and (r.detail.deal == "falling") is knife


def test_the_message_of_the_knife_says_what_fell():
    msg = [f for f in run(ret=-0.52, dd=-0.57).findings if f.code == "falling_knife"][0].message
    assert "fell 52% in 21 sessions" in msg and "57% below its 63-session peak" in msg and "not a discount" in msg


# ---------------------------------------------------------------- the warnings

def test_a_rich_price_and_an_unassessable_deal_are_warnings():
    assert codes(run(price=115.0), "warning") == ["rich_price"] and run(price=115.0).detail.deal == "rich"
    r = run(fetched={**metrics(), "sma_126": {"value": None, "note": "needs 126 bars, 17 stored"}})
    [f] = [f for f in r.findings if f.code == "value_not_assessable"]
    assert f.severity == "warning" and f.requires_human and "17 stored" in f.message and r.detail.deal == "unknown" and r.discount is None


def test_a_measure_that_cannot_be_made_counts_the_unknown_credit_and_is_listed():
    r = run(fetched={**metrics(), "sma_126": {"value": None}, "high_126d": {"value": None}})
    by = {m.name: m for m in r.measures}
    assert by["discount_to_avg_6m"].available is False and by["discount_to_avg_6m"].value == 0.5 and by["reward_to_risk"].available is False and by["earnings_yield"].available and "expected_excess" not in by
    assert r.score == pytest.approx(0.5 * 50 + 0.5 * 30 + 15.79, abs=0.01)


def test_no_metrics_at_all_is_not_assessable_but_does_not_raise():
    r = vl.assess(rec(), None, 88.0, V, CREDIT)
    assert "value_not_assessable" in codes(r) and r.detail.deal == "unknown"


def test_a_reduction_is_not_judged():
    r = vl.assess(rec(**{"action.action": "reduce"}), metrics(), 88.0, V, CREDIT)
    assert r.applicable is False and r.findings == []


# ---------------------------------------------------------------- the bonus conditions (scoring) and the history warnings

def test_the_bonus_needs_no_accuracy_or_value_blocker_quality_and_a_passed_red_flag_gate():
    from critic_helpers import CONFIG as c
    from prometheus.agents.critic.common import blocker, warning
    ok = [warning("value", "rich_price", "x")]
    assert sc.great_deal_bonus_allowed(ok, 1.0, True, c.value.bonus.min_quality)
    assert not sc.great_deal_bonus_allowed([blocker("value", "falling_knife", "x")], 1.0, True, 0.7)
    assert not sc.great_deal_bonus_allowed([blocker("accuracy", "data_mismatch", "x")], 1.0, True, 0.7)
    assert sc.great_deal_bonus_allowed([blocker("timing", "market_closed", "x")], 1.0, True, 0.7)     # a closed market says nothing about the deal
    assert not sc.great_deal_bonus_allowed(ok, 0.69, True, 0.7) and not sc.great_deal_bonus_allowed(ok, 1.0, False, 0.7)


def test_a_recent_earlier_recommendation_at_a_much_lower_entry_is_a_warning():
    r = rec()
    earlier = {"symbol": "PG", "identity": {"record_id": "old1", "as_of_utc": "2026-10-20T00:00:00Z"}, "trade": {"entry_mid": 80.0}}
    [f] = vl.history_warnings(r, 88.0, MemoryLedger([earlier]), NOW.date(), V)
    assert f.code == "price_above_last_recommendation" and "80.00" in f.message and f.severity == "warning"
    assert vl.history_warnings(r, 83.0, MemoryLedger([earlier]), NOW.date(), V) == []                          # 3.75% higher: under the 5% line
    same = {**earlier, "identity": {**earlier["identity"], "record_id": r.identity.record_id}}
    assert vl.history_warnings(r, 88.0, MemoryLedger([same]), NOW.date(), V) == [] and vl.history_warnings(r, 88.0, None, NOW.date(), V) == []


def test_adding_far_above_what_was_paid_is_a_warning():
    r = rec(**{"action.action": "add", "action.cost_basis": 70.0})
    assert [f.code for f in vl.history_warnings(r, 88.0, None, NOW.date(), V)] == ["adding_above_cost"]
    assert vl.history_warnings(rec(**{"action.action": "add", "action.cost_basis": 85.0}), 88.0, None, NOW.date(), V) == []
    assert vl.history_warnings(r, None, None, NOW.date(), V) == []
