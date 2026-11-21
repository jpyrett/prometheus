"""Value: is the price a good deal? (Critic spec 8.5 and Appendix B.3.) The captures hold no six-month average, so the worked examples use invented INPUTS (said so in the names): the rules are what is tested."""
from advisor_helpers import MemoryLedger
from critic_helpers import CONFIG, NOW, captured, rec_of, tamper

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


# ---------------------------------------------------------------- the blockers


def test_an_extreme_valuation_is_a_blocker_by_pe_and_by_sales_when_there_is_no_profit():
    assert "valuation_extreme" in codes(run(ey=0.012), "blocker") and "valuation_extreme" not in codes(run(ey=0.0125))
    r = run(eps=-1.0, ps=25.0)
    assert "valuation_extreme" in codes(r, "blocker") and "no profit" in [f for f in r.findings if f.code == "valuation_extreme"][0].message
    assert "valuation_extreme" not in codes(run(eps=-1.0, ps=20.0)) and "valuation_extreme" not in codes(run(eps=-1.0, ps=None))


# ---------------------------------------------------------------- the bonus conditions (scoring) and the history warnings


def test_a_recent_earlier_recommendation_at_a_much_lower_entry_is_a_warning():
    r = rec()
    earlier = {"symbol": "PG", "identity": {"record_id": "old1", "as_of_utc": "2026-10-20T00:00:00Z"}, "trade": {"entry_mid": 80.0}}
    [f] = vl.history_warnings(r, 88.0, MemoryLedger([earlier]), NOW.date(), V)
    assert f.code == "price_above_last_recommendation" and "80.00" in f.message and f.severity == "warning"
    assert vl.history_warnings(r, 83.0, MemoryLedger([earlier]), NOW.date(), V) == []                          # 3.75% higher: under the 5% line
    same = {**earlier, "identity": {**earlier["identity"], "record_id": r.identity.record_id}}
    assert vl.history_warnings(r, 88.0, MemoryLedger([same]), NOW.date(), V) == [] and vl.history_warnings(r, 88.0, None, NOW.date(), V) == []
