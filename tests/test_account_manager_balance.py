"""Spreading the money evenly over the purchases (design choices entries 78 and 80): equal DOLLARS for each company, whole shares, and a company whose even share buys less than
one share is dropped from the split. Since entry 80 this is the Account Manager's sizing (`sizing.size_purchases(..., allocation="even")`, which replaced `decisions.balance_evenly`), reached through
`decisions.decide_trades(..., balance="even")`; the tests keep what the old tests of `balance_evenly` proved.

REAL DATA ONLY. The prices are the real closing prices of real symbols (PG, MSFT, TSLA, GM on 2026-11-13) read from the real stored daily bars; there is no limit of a purchase's own any more (entry 91). The split through `decide_trades` runs over the Critic's REAL reviews of a real session (`tests/fixtures/session_real/bob_jones_oil.json`, migrated one time to take out the fields the Critic and the Advisor no longer have, every score, price and decision as recorded; see tests/test_session_replay.py: XOM and EOG approved, APA
denied). Nothing is invented, and no agent is replaced by a stand-in. The tests are skipped when the bars are not on the machine. The same split in more detail, and the round robin, are in
`tests/test_account_manager_sizing.py` and `tests/test_session_replay.py`.
"""
import copy
import json
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from prometheus.agents.account_manager import decisions
from prometheus.agents.account_manager import sizing as sz
from prometheus.common import work_order as wo
from prometheus.common.money import price as price_door
from prometheus.common.portfolio import Portfolio, Position
from prometheus.common.user_profile import UserProfile
from prometheus.tools.data_clerk import DataClerkTools

DAY = date(2026, 11, 13)
CAPTURE = Path(__file__).parent / "fixtures" / "session_real" / "bob_jones_oil.json"
MOMENT = datetime(2026, 11, 13, 20, 30, tzinfo=timezone.utc)          # 15:30 Eastern on Friday 2026-11-13: the simulated moment of the real run
PROFILE = UserProfile(user="Bob Jones", risk_tolerance="conservative", age=60, trade_mode="auto")
BOB = Portfolio(user="Bob Jones", cash_avail=Decimal("15000"))


@pytest.fixture(scope="module")
def closes():
    try:
        found = DataClerkTools().daily_bars(["PG", "MSFT", "TSLA", "GM"], DAY, DAY)["bars"]
    except FileNotFoundError:
        pytest.skip("the stored daily bars are not on this machine")
    if not all(found.get(s) for s in ("PG", "MSFT", "TSLA", "GM")):
        pytest.skip(f"the stored bars do not hold {DAY} for PG, MSFT, TSLA and GM")
    return {s: price_door(found[s][-1]["close"]) for s in found}                  # a stored bar is a float: it comes in through the price door, once


def split(symbols, closes, cash, **options):
    """{symbol: Sized} of an even split of `cash` over real purchases ranked in the order given, of a $15,000 account."""
    purchases = [sz.Purchase(record_id=f"rec-{s}", symbol=s, action="open", price=closes[s], rank=n) for n, s in enumerate(symbols)]
    return {s.symbol: s for s in sz.size_purchases(purchases, cash, allocation="even", **options)}


@pytest.fixture(scope="module")
def capture():
    return json.loads(CAPTURE.read_text(encoding="utf-8"))


def reviewed(capture) -> wo.WorkOrder:
    """The real work order as it was when the Critic had finished (the real recommendations and reviews), before the Account Manager decided."""
    order = wo.WorkOrder.model_validate(copy.deepcopy(capture["order"]))
    order.status, order.trade_decisions, order.trades, order.trader_runs = "reviewed", [], [], []
    return order


def decide(capture, order=None, portfolio=BOB, profile=PROFILE, balance="even", budget=None, mode="auto"):
    return decisions.decide_trades(order or reviewed(capture), portfolio, profile, None, MOMENT, mode, balance, budget)


# ---------------------------------------------------------------- through decide_trades, over the real reviews of the real session


def test_only_a_purchase_is_part_of_the_split_a_recommended_sale_is_never_traded_and_a_denial_is_not_part_of_anything(capture):
    """AMENDED (entry 104). EOG's real approved review is made a `reduce` here: it used to be the whole holding, asked about even in Auto mode (`pending_user`, 5 shares). A sale never comes from a recommendation now, so it has
    nothing to trade: `no_funds`, no shares, no question, and it is not part of the split, which the one purchase left (XOM) gets whole."""
    order = reviewed(capture)
    order.reviews[2]["action"] = "reduce"                                                              # EOG becomes a recommended sale of what the account holds
    holds = Portfolio(user="Bob Jones", cash_avail=Decimal("15000"), positions=[Position(symbol="EOG", shares=5, cost_basis=Decimal("100"), acquired=date(2026, 1, 5))])
    out = decide(capture, order=order, portfolio=holds).trade_decisions
    xom, apa, eog = out
    assert (xom["will_trade"], xom["balance"]["of"]) == ("yes_auto", 1) and xom["shares"] >= 1 and xom["balance"]["target_dollars"] == Decimal("15000")       # the one purchase that is part of the split gets the whole cash: no limit holds it back (entry 91)
    assert (eog["action"], eog["shares"], eog["est_cost"], eog["will_trade"], eog["size"]) == ("reduce", 0, 0, "no_funds", None) and "balance" not in eog and "is never traded" in eog["reason"]
    assert (apa["will_trade"], apa["shares"]) == ("no_system_denied", 0)
