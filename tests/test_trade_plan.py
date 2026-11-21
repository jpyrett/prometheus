"""Tests for prometheus.agents.advisor.trade_plan (Advisor v1 spec section 10, phase 5; design choices entry 80). The prices are literals that make the arithmetic checkable by hand.

The plan has NO SIZE since entry 80 (how much to buy or sell is the Account Manager's): the tests that were about the risk budget, the position and liquidity caps, the regime halving the budget,
the account's equity and the funds check were removed with the code they tested. The arithmetic of every cap, of the equity and of the funds check is tested on the Account Manager's code
(tests/test_account_manager_sizing.py); the prices of the plan against the real captures are tested in tests/test_trade_plan_real.py.
"""

from advisor_helpers import CONFIG, TODAY, fr, healthy

from prometheus.agents.advisor import trade_plan as tp
from prometheus.agents.advisor.record import TradePlan


def metrics(**kw):
    return healthy(**{"atr_14": 4.0, "last_close": 100.0, **kw})


def plan(family="capital_return", m=None, edge=0.004, horizon=126, quote=None, **kw):
    return tp.build(fr(family, edge=edge, horizon_bars=horizon), m if m is not None else metrics(), quote, TODAY, CONFIG, symbol="AAA", **kw)


def test_a_record_made_before_the_exit_conventions_has_a_fixed_stop_and_loads():
    """A plan record of Advisor config version 2 (no `trailing`, no `trail_distance`) is read as a fixed-stop plan with its target: both new fields have their old-record defaults."""
    old = TradePlan(entry_basis="last_close", family="trend", entry_mid=100.0, stop_distance=12.0, stop_price=88.0, target_price=101.8)
    assert old.trailing is False and old.trail_distance is None and old.target_price == 101.8


