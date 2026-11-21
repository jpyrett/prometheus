"""AN ORDER IN DOLLARS: a purchase the Account Manager sized from money is permitted as "at most $X of S", and the Account Manager turns the dollars into shares at the price the trade fills at (design choices entry 111, M2;
design/dollar_orders_spec.md section 4).

Jacob (2026-11-15): *"Yes, I want #2 built."* The decision keeps `shares`, `est_cost` and `price` as the ESTIMATE at the price it was sized at and adds `dollars`; the handshake approves `shares.buys(min(dollars, what is left of the
pot, the cash), the fill price)`, FEWER shares than the estimate when the price is higher and MORE when it is lower; the broker fills the shares approved; `verify_fill` checks the AMOUNT against the dollars and not the shares against
the estimate.

REAL DATA ONLY: the real oil session (`auto_oil_no_second_mark.json`: the Advisor's, the Critic's and the Account Manager's own records) decided again by the real Account Manager, the real Trader with no model, and the next
session's REAL OPENS read from the stored daily bars through the real Data Clerk: the decisions are sized at the close of Friday 2026-11-13 and traded in `backtest` mode as of Thursday evening, so the entry bar is Friday's:
EOG, CVX and CNQ open BELOW the price they were sized at (more shares for the dollars) and OXY opens ABOVE it (fewer). The real bob_jones_oil session gives the whole-share cut at the real open of XOM.
"""
import copy
import json
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from handshake_helpers import BOB, FRIDAY, MOMENT, PROFILE, close_of, friday_close, manager, trader  # noqa: F401  (friday_close is a fixture)
from test_neutral_recommendation import OIL_CASH, OIL_USER, account, allowed, oil_order  # noqa: F401  (allowed is a fixture)

from prometheus.agents.account_manager import decisions
from prometheus.agents.account_manager.handshake import decide_approval
from prometheus.agents.trader import tickets as tk
from prometheus.agents.trader.config import load_config
from prometheus.common import shares as sh
from prometheus.common import work_order as wo
from prometheus.common.money import money, price
from prometheus.common.portfolio import trade_amount

pytestmark = pytest.mark.usefixtures("friday_close")

EVENING = datetime(2026, 11, 12, 20, 30, tzinfo=timezone.utc)      # Thursday evening: the next session is Friday 2026-11-13, whose real opens are stored
OPENS = {"EOG": Decimal("157.25"), "CVX": Decimal("218.275"), "CNQ": Decimal("49.48"), "OXY": Decimal("59.00"), "XOM": Decimal("171.43")}      # the real opens of 2026-11-13 (the Data Clerk reads them from the stored day file; the test checks it)


def oil_four(**options):
    """The real oil session decided again over four companies, evenly: EOG, CVX (approved) and CNQ, OXY (neutral): $3,750.00 each of the $15,000.00."""
    return account(profile=OIL_USER, fractional_even_split=options.pop("fractional_even_split", True)).decide_trades(oil_order(), OIL_CASH, None, MOMENT, "auto", "even", None, count=4)


def backtest(order, portfolio=OIL_CASH, profile=OIL_USER):
    account(profile=profile).perform_trades(order, portfolio, trader(mode="backtest"), EVENING)
    return {t["symbol"]: t for t in order.trades}


# ---------------------------------------------------------------- the decision


def test_a_decision_whose_dollars_are_not_a_number_to_the_cent_or_pass_the_pot_is_not_permitted_by_the_program(allowed):
    out = oil_four()
    record = out.trade_decisions[0]["record_id"]
    for bad, why in ((0, "above zero"), (Decimal("-5.0"), "above zero"), ("3750", "above zero"), (True, "above zero"), (Decimal("3750.001"), "to the cent"), (Decimal("NaN"), "above zero")):
        broken = copy.deepcopy(out)
        broken.trade_decisions[0]["dollars"] = bad
        assert any("the decision's dollars must be a number above zero to the cent" in r for r in wo.verify_trade(broken, record, MOMENT)), (bad, why)
    broken = copy.deepcopy(out)
    broken.trade_decisions[0]["dollars"] = Decimal("15000.01")
    assert any("are more than the money of the request, $15,000.00" in r for r in wo.verify_trade(broken, record, MOMENT))
    sale = copy.deepcopy(out)
    sale.trade_decisions[0]["action"] = "reduce"
    assert any("a sale is an order in shares" in r for r in wo.verify_trade(sale, record, MOMENT))


# ---------------------------------------------------------------- the handshake turns the dollars into shares at the fill price

def ticket_of(order, symbol):
    return next(t for t in tk.build_tickets(order, OIL_CASH, MOMENT, load_config(), "backtest") if t.symbol == symbol).model_dump(mode="json")


def test_the_dollars_buy_fewer_shares_at_a_higher_price_more_at_a_lower_one_and_never_cost_more_than_the_dollars(allowed):
    out = oil_four()
    ticket = ticket_of(out, "OXY")
    assert money(ticket["dollars"]) == Decimal("3750.00") and sh.quantity(ticket["shares_approved"]) == sh.quantity(ticket["shares"]) == out.trade_decisions[3]["shares"] if out.trade_decisions[3]["symbol"] == "OXY" else True      # the ticket is a dict in JSON mode: its numbers go through the doors
    same = decide_approval(OIL_CASH, ticket, Decimal("58.75"), FRIDAY, dollars=Decimal("3750.00"))
    assert same.approved and same.shares == sh.quantity(ticket["shares"]) and trade_amount(same.shares, Decimal("58.75")) == Decimal("3750.00")                                       # at the sizing price: the estimate
    for fill_price in (Decimal("59.00"), Decimal("61.37"), Decimal("58.00"), Decimal("40.00"), Decimal("58.7501")):
        answer = decide_approval(OIL_CASH, ticket, fill_price, FRIDAY, dollars=Decimal("3750.00"))
        assert answer.approved and answer.shares == sh.buys(Decimal("3750.00"), fill_price) and trade_amount(answer.shares, fill_price) == Decimal("3750.00") and (answer.shares < sh.quantity(ticket["shares"])) == (fill_price > Decimal("58.75"))
    cash = decide_approval(OIL_CASH.model_copy(update={"cash_avail": Decimal("1000.00")}), ticket, Decimal("59.00"), FRIDAY, dollars=Decimal("3750.00"))
    assert cash.shares == sh.buys(Decimal("1000.00"), Decimal("59.00")) and "held by the cash to $1,000.00" in cash.reason
    pot = decide_approval(OIL_CASH, ticket, Decimal("59.00"), FRIDAY, spendable=Decimal("500.00"), pot=Decimal("15000.00"), dollars=Decimal("3750.00"))
    assert pot.shares == sh.buys(Decimal("500.00"), Decimal("59.00")) and "held by the money of this request to $500.00" in pot.reason
    none = decide_approval(OIL_CASH, ticket, Decimal("59.00"), FRIDAY, spendable=Decimal("0.00"), pot=Decimal("15000.00"), dollars=Decimal("3750.00"))
    assert (none.approved, none.code) == (False, "insufficient_cash") and "the money of this request is used up" in none.reason and "is not the limit" in none.reason
    assert decide_approval(OIL_CASH, {**ticket, "shares": 1}, Decimal("59.00"), FRIDAY).shares == 1                                                          # no dollars: an order in shares, asked and allowed as ever (the estimate caps it)


# ---------------------------------------------------------------- the program checks the fill against the dollars, not the shares against the estimate

def test_verify_fill_checks_the_amount_of_an_order_in_dollars_and_the_shares_of_an_order_in_shares(allowed):
    order = oil_four()
    t = backtest(order)
    trades = order.trades
    first = copy.deepcopy(trades[0])
    assert first["fill"]["shares"] > first["shares_approved"] and wo.verify_fill(order, first, trades) == []      # MORE shares than the estimate is fine for an order in dollars: its shares are an estimate
    over = copy.deepcopy(first)
    over["fill"]["shares"] = sh.plus(over["fill"]["shares"], Decimal("0.01"))                                              # a hundredth of a share more costs more than the $3,750.00
    over["fill"]["amount"] = trade_amount(over["fill"]["shares"], over["fill"]["price"])
    reasons = wo.verify_fill(order, over, trades)
    assert any("cost $3,751.57 but the decision's dollars are $3,750.00" in r for r in reasons) and not any("shares but the decision gave" in r for r in reasons)
    understated = copy.deepcopy(first)
    understated["fill"]["amount"] = Decimal("3800.00")                                                                      # the fill says it cost more than the dollars
    assert any("says it cost $3,800.00 but the decision's dollars are $3,750.00" in r for r in wo.verify_fill(order, understated, trades))
    shares_order = copy.deepcopy(order)
    for d in shares_order.trade_decisions:
        d.pop("dollars", None)
    assert any("shares but the decision gave" in r for r in wo.verify_fill(shares_order, first, trades))         # the same fill against an order in shares: the old check, exactly as it was
    assert wo.verify_trades(order) == [] and wo.verify_fill(order, first, trades) == []
