"""Spreading the money evenly over the purchases (design choices entries 78 and 80): equal DOLLARS for each company, whole shares, never more than a company's own limits, and a company whose even share buys less than
one share is dropped from the split. Since entry 80 this is the Account Manager's sizing (`sizing.size_purchases(..., allocation="even")`, which replaced `decisions.balance_evenly`), reached through
`decisions.decide_trades(..., balance="even")`; the tests keep what the old tests of `balance_evenly` proved.

REAL DATA ONLY. The prices are the real closing prices of real symbols (PG, MSFT, TSLA, GM on 2026-11-13) read from the real stored daily bars; a purchase's own limit is the profile's position limit (10% of a $15,000
account, whole shares) worked out from those prices. The split through `decide_trades` runs over the Critic's REAL reviews of a real session (`tests/fixtures/session_real/bob_jones_oil.json`, migrated one time to take out the fields the Critic and the Advisor no longer have, every score, price and decision as recorded; see tests/test_session_replay.py: XOM and EOG approved, APA
denied). Nothing is invented, and no agent is replaced by a stand-in. The tests are skipped when the bars are not on the machine. The same split in more detail, and the round robin, are in
`tests/test_account_manager_sizing.py` and `tests/test_session_replay.py`.
"""
import copy
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from prometheus.agents.account_manager import decisions
from prometheus.agents.account_manager import sizing as sz
from prometheus.common import work_order as wo
from prometheus.common.portfolio import Portfolio, Position
from prometheus.common.user_profile import UserProfile
from prometheus.tools.data_clerk import DataClerkTools

DAY = date(2026, 11, 13)
EQUITY, POSITION_LIMIT = 15000.0, 0.10
CAPTURE = Path(__file__).parent / "fixtures" / "session_real" / "bob_jones_oil.json"
MOMENT = datetime(2026, 11, 13, 20, 30, tzinfo=timezone.utc)          # 15:30 Eastern on Friday 2026-11-13: the simulated moment of the real run
PROFILE = UserProfile(user="Bob Jones", risk_tolerance="conservative", age=60, trade_mode="auto", auto_min_score=70)
WIDE = PROFILE.model_copy(update={"max_position_pct": 1.0, "max_new_exposure_pct": 1.0, "max_loss_per_trade_pct": 0.05})          # a user whose own limits are not what holds a purchase back: only the money is
BOB = Portfolio(user="Bob Jones", cash_avail=15000.0)


@pytest.fixture(scope="module")
def closes():
    try:
        found = DataClerkTools().daily_bars(["PG", "MSFT", "TSLA", "GM"], DAY, DAY)["bars"]
    except FileNotFoundError:
        pytest.skip("the stored daily bars are not on this machine")
    if not all(found.get(s) for s in ("PG", "MSFT", "TSLA", "GM")):
        pytest.skip(f"the stored bars do not hold {DAY} for PG, MSFT, TSLA and GM")
    return {s: found[s][-1]["close"] for s in found}


def own_shares(price: float) -> int:
    """What a 10% position limit lets a $15,000 account buy of one stock, in whole shares: a purchase's own limit."""
    return math.floor(EQUITY * POSITION_LIMIT / price)


def split(symbols, closes, cash, **options):
    """{symbol: Sized} of an even split of `cash` over real purchases ranked in the order given, over the profile's own limits and a $15,000 account."""
    purchases = [sz.Purchase(record_id=f"rec-{s}", symbol=s, action="open", price=closes[s], rank=n) for n, s in enumerate(symbols)]
    return {s.symbol: s for s in sz.size_purchases(purchases, EQUITY, cash, PROFILE.sizing_limits(), allocation="even", **options)}


@pytest.fixture(scope="module")
def capture():
    return json.loads(CAPTURE.read_text(encoding="utf-8"))


def reviewed(capture) -> wo.WorkOrder:
    """The real work order as it was when the Critic had finished (the real recommendations and reviews), before the Account Manager decided."""
    order = wo.WorkOrder.model_validate(copy.deepcopy(capture["order"]))
    order.status, order.trade_decisions, order.trades, order.trader_runs = "reviewed", [], [], []
    return order


def decide(capture, order=None, portfolio=BOB, profile=PROFILE, balance="even", budget=None, mode="auto"):
    return decisions.decide_trades(order or reviewed(capture), portfolio, profile, 80, None, MOMENT, mode, balance, budget)


# ---------------------------------------------------------------- the rule on its own, over real prices

def test_every_company_gets_the_same_dollars_whatever_its_price(closes):
    got = split(["PG", "MSFT", "GM"], closes, 4500.0)
    smallest = min(own_shares(closes[s]) * closes[s] for s in got)                                # the smallest own limit, in dollars, rules: no company is given more than its own limit
    assert all(s.target_dollars == round(smallest, 2) and s.of == 3 and s.allocation == "even" for s in got.values())
    for symbol, s in got.items():
        assert s.shares == math.floor(smallest / closes[symbol] + 1e-9) >= 1 and s.est_cost <= smallest + 1e-6 and smallest - s.est_cost < closes[symbol]        # whole shares: under the target by less than one share
        assert s.shares <= own_shares(closes[symbol]) and "even split" in s.notes[-1]


def test_the_money_can_be_what_limits_the_split(closes):
    got = split(["GM", "PG"], closes, 400.0)                                                       # $200 each: two shares of GM, one of PG
    assert got["GM"].target_dollars == got["PG"].target_dollars == 200.0
    assert [(s.symbol, s.shares) for s in got.values()] == [("GM", math.floor(200 / closes["GM"])), ("PG", math.floor(200 / closes["PG"]))]
    assert got["GM"].binding == got["PG"].binding == "money"


def test_a_company_that_could_not_buy_one_share_is_dropped_and_the_others_are_balanced_again(closes):
    got = split(["GM", "PG", "TSLA"], closes, 600.0)                                               # $200 each buys no share of TSLA: it is out, and $300 each is for the other two
    assert (got["TSLA"].shares, got["TSLA"].est_cost, got["TSLA"].binding) == (0, 0.0, "no_room") and f"does not buy one share at ${closes['TSLA']:,.2f}" in got["TSLA"].notes[-1]
    assert [got[s].shares for s in ("GM", "PG")] == [math.floor(300 / closes["GM"]), math.floor(300 / closes["PG"])]
    assert got["GM"].target_dollars == got["PG"].target_dollars == 300.0 and got["GM"].of == got["PG"].of == 2


def test_nothing_is_balanced_when_no_purchase_can_afford_a_share(closes):
    got = split(["TSLA", "MSFT"], closes, 100.0)
    assert all(s.shares == 0 and s.binding == "no_room" and s.est_cost == 0.0 for s in got.values())


def test_a_whole_number_of_shares_that_the_float_makes_a_hair_short_is_still_bought(closes):
    price = closes["GM"]
    got = split(["GM"], closes, round(price * 3, 2))                                               # exactly three shares of money, in cents
    assert got["GM"].shares == 3 and got["GM"].est_cost == round(price * 3, 2)


def test_a_company_is_never_given_more_than_its_own_limits_allow(closes):
    got = split(["GM", "PG"], closes, 1_000_000.0, budget=None)                                    # far more money than either limit: the smaller own limit, in dollars, is the target
    target = min(own_shares(closes["GM"]) * closes["GM"], own_shares(closes["PG"]) * closes["PG"])
    assert [s.shares for s in got.values()] == [math.floor(target / closes["GM"] + 1e-9), math.floor(target / closes["PG"] + 1e-9)]
    assert all(s.shares <= own_shares(closes[s.symbol]) for s in got.values())


# ---------------------------------------------------------------- through decide_trades, over the real reviews of the real session

def test_decide_trades_gives_xom_and_eog_the_same_dollars_and_cuts_the_dearer_one_to_match(capture):
    out = decide(capture).trade_decisions
    xom, apa, eog = out
    assert [d["will_trade"] for d in out] == ["yes_auto", "no_system_denied", "yes_auto"]
    assert (xom["shares"], eog["shares"]) == (8, 8) and xom["est_cost"] == 1364.72 and eog["est_cost"] == 1266.56      # the 10% limit alone would give EOG 9 shares: it is cut to the dollars of XOM's 8
    assert xom["balance"] == {"mode": "even", "target_dollars": 1364.72, "of": 2} == eog["balance"] and "balanced evenly" in xom["reason"]
    assert (xom["size"]["binding"], eog["size"]["binding"]) == ("position", "even_split") and xom["size"]["allocation"] == eog["size"]["allocation"] == "even"
    assert apa["shares"] == 0 and apa["size"] is None and "balance" not in apa                       # the denied company is not part of the split


def test_without_a_split_and_in_ask_mode_each_purchase_is_what_its_own_limits_allow(capture):
    out = decide(capture, balance=None, mode="ask").trade_decisions
    assert [(d["symbol"], d["shares"], d["will_trade"]) for d in out if d["symbol"] != "APA"] == [("XOM", 8, "pending_user"), ("EOG", 9, "pending_user")]
    assert not any("balance" in d for d in out) and {d["size"]["allocation"] for d in out if d["size"]} == {"as_much_as_possible"}


def test_the_money_can_be_what_limits_the_split_through_decide_trades(capture):
    poorer = Portfolio(user="Bob Jones", cash_avail=1000.0, positions=[Position(symbol="GM", shares=50, cost_basis=20.0, acquired=date(2026, 1, 5))])      # $1,000 in cash and $1,000 of GM at cost: an equity of $2,000
    out = [d for d in decide(capture, portfolio=poorer, profile=WIDE).trade_decisions if d["will_trade"] == "yes_auto"]               # the user's own limits are wide here, and the exposure room ($2,000) is over the cash: the cash is what limits it
    assert {d["balance"]["target_dollars"] for d in out} == {500.0} and [d["shares"] for d in out] == [math.floor(500 / 170.59), math.floor(500 / 158.32)] == [2, 3]
    assert all(d["size"]["binding"] == "money" for d in out)


def test_a_budget_in_the_users_words_caps_the_split_by_the_real_prices(capture):
    out = [d for d in decide(capture, budget=2000.0).trade_decisions if d["will_trade"] == "yes_auto"]
    assert {d["balance"]["target_dollars"] for d in out} == {1000.0} and [d["shares"] for d in out] == [math.floor(1000 / 170.59), math.floor(1000 / 158.32)] == [5, 6]
    assert sum(d["est_cost"] for d in out) <= 2000.0 and all(d["size"]["binding"] == "budget" for d in out)


def test_a_company_that_could_not_buy_one_share_is_dropped_through_decide_trades_and_is_not_asked_about(capture):
    asked = []
    poorer = Portfolio(user="Bob Jones", cash_avail=300.0)                                           # $150 each buys no share of either
    out = decisions.decide_trades(reviewed(capture), poorer, WIDE.model_copy(update={"trade_mode": "ask"}), 80, lambda card: asked.append(card["symbol"]) or True, MOMENT, "ask", "even", None)
    shares = {d["symbol"]: (d["shares"], d["will_trade"]) for d in out.trade_decisions}
    assert shares == {"XOM": (0, "no_funds"), "APA": (0, "no_system_denied"), "EOG": (0, "no_funds")} and asked == [] and out.status == "decided"
    assert all("does not buy one share" in d["reason"] for d in out.trade_decisions if d["will_trade"] == "no_funds") and not any("balance" in d for d in out.trade_decisions)


def test_only_a_purchase_is_part_of_the_split_a_sale_is_the_whole_holding_and_a_denial_is_not_part_of_anything(capture):
    order = reviewed(capture)
    order.reviews[2]["action"] = "reduce"                                                              # EOG becomes a sale of what the account holds
    holds = Portfolio(user="Bob Jones", cash_avail=15000.0, positions=[Position(symbol="EOG", shares=5, cost_basis=100.0, acquired=date(2026, 1, 5))])
    out = decide(capture, order=order, portfolio=holds).trade_decisions
    xom, apa, eog = out
    assert (xom["shares"], xom["will_trade"], xom["balance"]["of"]) == (9, "yes_auto", 1) and xom["balance"]["target_dollars"] == 1535.31       # the one purchase that is part of the split: 10% of the $15,500 equity (the 5 EOG shares at cost) is 9 shares
    assert (eog["action"], eog["shares"], eog["est_cost"], eog["will_trade"], eog["size"]) == ("reduce", 5, 0.0, "pending_user", None) and "balance" not in eog
    assert (apa["will_trade"], apa["shares"]) == ("no_system_denied", 0)


def test_no_purchase_that_trades_means_no_split(capture):
    order = reviewed(capture)
    order.reviews = [r for r in order.reviews if not r["approved"]]                                 # only APA, which the Critic denied
    out = decide(capture, order=order).trade_decisions
    assert [d["will_trade"] for d in out] == ["no_system_denied"] and out[0]["shares"] == 0 and "balance" not in out[0] and out[0]["size"] is None
