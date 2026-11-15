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


# ---------------------------------------------------------------- labels at their bounds

@pytest.mark.parametrize("premium, falling, label", [
    (-0.20, False, "great"), (-0.10, False, "great"), (-0.0999, False, "good"), (-0.03, False, "good"), (-0.0299, False, "fair"), (0.0, False, "fair"), (0.10, False, "fair"),
    (0.1001, False, "rich"), (0.30, False, "rich"), (0.3001, False, "overpriced"), (-0.5, True, "falling"), (0.5, True, "falling"), (None, False, "unknown"), (None, True, "falling")])
def test_the_label_is_the_first_that_fits(premium, falling, label):
    assert vl.label_of(premium, falling, V) == label


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


# ---------------------------------------------------------------- the warnings

def test_a_rich_price_and_an_unassessable_deal_are_warnings():
    assert codes(run(price=115.0), "warning") == ["rich_price"] and run(price=115.0).detail.deal == "rich"
    r = run(fetched={**metrics(), "sma_126": {"value": None, "note": "needs 126 bars, 17 stored"}})
    [f] = [f for f in r.findings if f.code == "value_not_assessable"]
    assert f.severity == "warning" and f.requires_human and "17 stored" in f.message and r.detail.deal == "unknown" and r.discount is None


def test_a_reduction_is_not_judged():
    r = vl.assess(rec(**{"action.action": "reduce"}), metrics(), 88.0, V, CREDIT)
    assert r.applicable is False and r.findings == []


# ---------------------------------------------------------------- the bonus conditions (scoring) and the history warnings


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
