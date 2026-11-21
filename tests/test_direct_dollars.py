"""A DIRECT ORDER WITH MONEY: as many whole shares as the money pays for, and a fraction of a share with what is left (design choices entry 111, M3; design/dollar_orders_spec.md section 5).

Jacob (2026-11-15): *"When a direct order is given, buy as many whole shares as possible and a fractional share for the leftover funds available."* A stock the user named with dollars, or with no number, is sized by the
Account Manager (`AccountManager._size_direct`): the money is spread as ever (round robin in whole shares, one share of each stock in turn), what it leaves buys a fraction of a share of the stock whose turn is next, and
every such purchase is an ORDER IN DOLLARS, so that at the fill each buys its dollars' worth. The recommendation path is not changed (its leftover stays cash); a number of shares the user typed or answered is an order in shares.

REAL DATA ONLY, as `test_direct_order.py` (whose helpers these tests use): the real model's recorded forms for "Buy $5,000 of MSFT" and the others (`direct_order_forms.json`), its recorded answer to "how much?"
(`clarify_ask_mode.json`), the stored closes of Friday 2026-11-13 and the real next open (Friday's, with the run as of Thursday evening), the real Trader with no model.
"""
from datetime import datetime
from decimal import Decimal

import pytest
from handshake_helpers import FRIDAY, MOMENT, close_of, friday_close, trader  # noqa: F401  (friday_close is a fixture)
from test_direct_order import BILL, BILL_PROFILE, account, direct_order, sized_order

from prometheus.common import work_order as wo
from prometheus.common.portfolio import Portfolio

pytestmark = pytest.mark.usefixtures("friday_close")


def test_a_number_of_shares_the_user_typed_or_answered_is_an_order_in_shares_as_it_was():
    typed, am = direct_order("buy", "MSFT", 30)
    d = typed.trade_decisions[0]
    assert "dollars" not in d and d["size"] is None and d["shares"] == 30 and isinstance(d["shares"], Decimal) and wo.verify_trade(typed, d["record_id"], MOMENT) == []
    am.perform_trades(typed, BILL, trader(), MOMENT)
    assert typed.trades[0]["dollars"] is None and typed.trades[0]["shares"] == 30
    answered, *_ = sized_order("I want to buy MSFT", mode="ask", answers="shares_each_raise_yes")                                      # "10 shares of each": whole shares, an order in shares
    e = answered.trade_decisions[0]
    assert e["shares"] == 10 and "dollars" not in e and e["size"]["pot"] == Decimal("20000") and answered.direct_orders[0]["how_many"] == "asked"
    typed.trade_decisions[0]["dollars"] = Decimal("100")
    assert any("an order in shares: it carries no dollars" in r for r in wo.verify_trade(typed, d["record_id"], MOMENT))                # the word dollars on a typed order permits nothing


