"""Spreading the money evenly over the purchases (design choices entries 78 and 80): equal DOLLARS for each company, whole shares, and a company whose even share buys less than
one share is dropped from the split. Since entry 80 this is the Account Manager's sizing (`sizing.size_purchases(..., allocation="even")`, which replaced `decisions.balance_evenly`), reached through
`decisions.decide_trades(..., balance="even")`; the tests keep what the old tests of `balance_evenly` proved.

REAL DATA ONLY. The prices are the real closing prices of real symbols (PG, MSFT, TSLA, GM on 2026-11-13) read from the real stored daily bars; there is no limit of a purchase's own any more (entry 91). The split through `decide_trades` runs over the Critic's REAL reviews of a real session (`tests/fixtures/session_real/bob_jones_oil.json`, migrated one time to take out the fields the Critic and the Advisor no longer have, every score, price and decision as recorded; see tests/test_session_replay.py: XOM and EOG approved, APA
denied). Nothing is invented, and no agent is replaced by a stand-in. The tests are skipped when the bars are not on the machine. The same split in more detail, and the round robin, are in
`tests/test_account_manager_sizing.py` and `tests/test_session_replay.py`.
"""
import copy
import json
import math
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
EQUITY = Decimal("15000")
CAPTURE = Path(__file__).parent / "fixtures" / "session_real" / "bob_jones_oil.json"
MOMENT = datetime(2026, 11, 13, 20, 30, tzinfo=timezone.utc)          # 15:30 Eastern on Friday 2026-11-13: the simulated moment of the real run
PROFILE = UserProfile(user="Bob Jones", risk_tolerance="conservative", age=60, trade_mode="auto")
WIDE = PROFILE                                                                                                                         # there is no limit of the profile any more (entry 91): only the money holds a purchase back
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


# ---------------------------------------------------------------- the rule on its own, over real prices

def test_every_company_gets_the_same_dollars_whatever_its_price(closes):
    got = split(["PG", "MSFT", "GM"], closes, Decimal("4500"))
    target = Decimal("1500.00")                                                                  # no limit of any company's own: the money divided by the companies is the target
    assert all(s.target_dollars == target and s.of == 3 and s.allocation == "even" for s in got.values())
    for symbol, s in got.items():
        assert s.shares >= math.floor(target / closes[symbol]) >= 1 and "even split" in s.notes[0]                  # AMENDED (entry 108): the equal amount buys these whole shares, and the leftover may buy more (below)
    assert {s: got[s].shares for s in got} == {"PG": 8, "MSFT": 3, "GM": 18} and Decimal("4500") - sum(s.est_cost for s in got.values()) == Decimal("21.96")      # by hand: step 1 PG 8, MSFT 2, GM 17 ($3,868.16); the $631.84 left buys MSFT 1 and GM 1 more


def test_a_company_that_could_not_buy_one_share_is_dropped_and_the_others_are_balanced_again(closes):
    got = split(["GM", "PG", "TSLA"], closes, Decimal("600"))                                               # $200 each buys no share of TSLA: it is out, and $300 each is for the other two
    assert (got["TSLA"].shares, got["TSLA"].est_cost, got["TSLA"].binding) == (0, 0, "no_room") and f"does not buy one share at ${closes['TSLA']:,.2f}" in got["TSLA"].notes[-1]
    assert [got[s].shares for s in ("GM", "PG")] == [3, 2]                                         # AMENDED (entry 108): $300 each bought GM 3 ($258.00) and PG 1 ($169.80); the $172.20 left bought PG a second share ($339.60), and GM's next ($86.00) no longer fits the $2.40 left
    assert got["GM"].target_dollars == got["PG"].target_dollars == Decimal("300") and got["GM"].of == got["PG"].of == 2


# ---------------------------------------------------------------- through decide_trades, over the real reviews of the real session


def test_a_company_that_could_not_buy_one_share_is_dropped_through_decide_trades_and_is_not_asked_about(capture):
    asked = []
    poorer = Portfolio(user="Bob Jones", cash_avail=Decimal("300"))                                           # $150 each buys no share of either
    out = decisions.decide_trades(reviewed(capture), poorer, WIDE.model_copy(update={"trade_mode": "ask"}), lambda card: asked.append(card["symbol"]) or True, MOMENT, "ask", "even", None)
    shares = {d["symbol"]: (d["shares"], d["will_trade"]) for d in out.trade_decisions}
    assert shares == {"XOM": (0, "no_funds"), "APA": (0, "no_system_denied"), "EOG": (0, "no_funds")} and asked == [] and out.status == "decided"
    assert all("does not buy one share" in d["reason"] for d in out.trade_decisions if d["will_trade"] == "no_funds") and not any("balance" in d for d in out.trade_decisions)


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
