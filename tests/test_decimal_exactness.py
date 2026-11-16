"""The identities that floats only held to a tolerance, now exact (design choices entry 112; design/decimal_money_spec.md section 7), on REAL data: the real captures of `tests/fixtures/session_real/`, the stored closes of Friday 2026-11-13
through the real Data Clerk and the real Trader with no model (`handshake_helpers.py`). No stand-in agent, no stand-in model.

Run:  pytest tests/test_decimal_exactness.py -q

    * the cash after a run is the cash before less the sum of the fills, exactly;
    * an even split's amounts add up to the pot exactly, and `est_cost` is `trade_amount(shares, price)` exactly;
    * a work order with Decimal values is written as numbers and read back as the same numbers; a capture made BEFORE this change loads and writes the same numbers; a Decimal with more digits than a float holds is refused, not shortened.
"""
import copy
import json
from decimal import Decimal
from pathlib import Path

import pytest
from handshake_helpers import BOB, MOMENT, PROFILE, friday_close, manager, real_order, trader  # noqa: F401  (friday_close is a fixture)

from prometheus.agents.account_manager import decisions
from prometheus.agents.trader.record import Trade
from prometheus.common import shares as sh
from prometheus.common import work_order as wo
from prometheus.common.money import dumps, money, price
from prometheus.common.portfolio import trade_amount

FIXTURES = Path(__file__).parent / "fixtures" / "session_real"
CAPTURES = ["auto_even_split_dollars", "auto_even_split_fractions", "bob_jones_oil", "bob_jones_oil_round_robin", "auto_oil_no_second_mark", "pam_smith_neutral", "bill_gates_direct_order"]


def capture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def reviewed() -> wo.WorkOrder:
    """The real session's order (`bob_jones_oil.json`) as the Critic left it, before the Account Manager decides: the real recommendations and reviews, no decision, no trade."""
    order = wo.WorkOrder.model_validate(copy.deepcopy(capture("bob_jones_oil")["order"]))
    order.status, order.trade_decisions, order.trades, order.trader_runs = "reviewed", [], [], []
    return order


# ---------------------------------------------------------------- the same numbers, written and read back

@pytest.mark.parametrize("name", CAPTURES)
def test_a_capture_made_before_this_change_loads_and_its_trades_are_written_as_the_same_numbers(name):
    """The trades of a recorded session are floats once `json` has read them; each goes through the typed record (Decimals) and comes out as the same JSON numbers, a whole value as an integer: nothing is lost and nothing is a string."""
    order = capture(name)["order"]
    wo.WorkOrder.model_validate(order)                                       # it loads
    for recorded in order["trades"]:
        written = Trade.model_validate(recorded).model_dump(mode="json")
        assert {key: written[key] for key in recorded} == recorded, f"{name} {recorded['trade_id']}"
        text = dumps(Trade.model_validate(recorded).model_dump())
        assert not any(isinstance(v, str) and v.replace(".", "").isdigit() for v in json.loads(text).values())      # a number is never a string


def test_a_work_order_with_decimal_decisions_is_written_as_numbers_and_read_back_as_the_same_money_prices_and_shares():
    order = decisions.decide_trades(reviewed(), BOB, PROFILE, None, MOMENT, "auto", "even", Decimal("15000.00"), fractional=True)       # the real reviews, decided again: fractional shares, orders in dollars
    made = [d for d in order.trade_decisions if d["will_trade"] == "yes_auto"]
    assert made and all(isinstance(d["shares"], Decimal) and isinstance(d["est_cost"], Decimal) and isinstance(d["dollars"], Decimal) for d in made)
    text = order.model_dump_json()
    read = wo.WorkOrder.model_validate_json(text)                            # the floats of the JSON are in the decisions now
    assert all(isinstance(d["est_cost"], (int, float)) for d in read.trade_decisions if d["will_trade"] == "yes_auto")
    for before, after in zip(order.trade_decisions, read.trade_decisions):
        if before["will_trade"] != "yes_auto":
            continue
        assert sh.quantity(after["shares"]) == before["shares"] and money(after["est_cost"]) == before["est_cost"] and money(after["dollars"]) == before["dollars"] and price(after["price"]) == before["price"]
        assert money(after["size"]["pot"]) == before["size"]["pot"] and sh.quantity(after["size"]["shares"]) == before["size"]["shares"]
    assert '"est_cost":"' not in text and '"shares":"' not in text and '"dollars":"' not in text and '"pot":"' not in text      # numbers, not strings


def test_a_decimal_with_more_digits_than_a_float_holds_is_refused_by_the_encoder_and_never_shortened():
    order = real_order()
    order.trade_decisions[0]["est_cost"] = Decimal("5000.1234567890123456789")
    with pytest.raises(ValueError):
        order.model_dump_json()
    with pytest.raises(ValueError):
        dumps(order.model_dump())


# ---------------------------------------------------------------- exact identities

def test_an_even_split_adds_up_to_the_pot_exactly_and_each_cost_is_its_shares_times_its_price_to_the_cent():
    order = decisions.decide_trades(reviewed(), BOB, PROFILE, None, MOMENT, "auto", "even", Decimal("14999.99"), fractional=True)
    made = [d for d in order.trade_decisions if d["will_trade"] == "yes_auto"]
    pot = {d["size"]["pot"] for d in made}
    assert pot == {Decimal("14999.99")} and len(made) >= 2
    assert sum((d["dollars"] for d in made), Decimal("0.00")) == pot.pop()                                  # the amounts of an even split are the pot, to the cent: the odd cent went to the first purchase
    for d in made:
        assert d["est_cost"] == trade_amount(d["shares"], d["price"]) == d["dollars"]                      # est_cost is trade_amount(shares, price), exactly


def test_a_whole_share_split_costs_exactly_trade_amount_and_never_more_than_the_pot():
    order = real_order()
    made = [d for d in order.trade_decisions if d["will_trade"] == "yes_auto"]
    review = {r["record_id"]: r for r in order.reviews}
    for d in made:
        at = price(review[d["record_id"]]["timing"]["deal_price"])
        assert d["est_cost"] == trade_amount(d["shares"], at) and sh.whole(d["shares"])
    assert sum((d["est_cost"] for d in made), Decimal("0.00")) <= made[0]["size"]["pot"]


def test_the_cash_after_a_run_is_the_cash_before_less_the_sum_of_the_fills_exactly(friday_close):
    order = real_order()
    am = manager()
    am.perform_trades(order, BOB, trader(), MOMENT)
    run = order.trader_runs[-1]
    fills = [t["fill"] for t in order.trades if t["status"] == "filled"]
    assert len(fills) >= 2
    assert all(isinstance(f["amount"], Decimal) and f["amount"] == trade_amount(f["shares"], f["price"]) for f in fills)            # each fill's amount is shares times price, to the cent
    assert run["portfolio_before"]["cash"] == BOB.cash_avail
    assert run["portfolio_after"]["cash"] == BOB.cash_avail - sum((f["amount"] for f in fills), Decimal("0.00"))           # exactly, with no tolerance
    assert am.current_portfolio().cash_avail == run["portfolio_after"]["cash"]


# ---------------------------------------------------------------- the account book and the working portfolio agree to the cent, after every fill

from test_trader_handshake import db, store  # noqa: E402,F401  (the real local Postgres on THROWAWAY accounts, deleted at the end, and a check that the three samples did not move)


def test_the_books_cash_is_the_cash_before_less_the_fills_so_far_exactly_after_every_fill_and_the_working_portfolio_agrees(db, monkeypatch, friday_close):
    account_id = db.account("15000.00")
    am, order = db.manager(account_id), real_order()
    book_after_each_fill = []
    real = am.record_fill

    def watched(trade):
        booking = real(trade)
        book_after_each_fill.append((trade["fill"]["amount"], booking.cash_avail, db.cash(account_id)))      # the Account Manager's answer, and the book read on its own
        return booking

    monkeypatch.setattr(am, "record_fill", watched)
    am.perform_trades(order, None, trader(), MOMENT, book=True)
    assert len(book_after_each_fill) >= 2
    cash = Decimal("15000.00")
    for amount, said, read in book_after_each_fill:
        cash -= amount
        assert said == read == cash and all(isinstance(x, Decimal) for x in (said, read))                    # exactly, to the cent, with no tolerance
    assert am.current_portfolio().cash_avail == db.cash(account_id) == order.trader_runs[-1]["portfolio_after"]["cash"] == cash
    assert sum((t["fill"]["amount"] for t in order.trades if t["status"] == "filled"), Decimal("0.00")) == Decimal("15000.00") - cash
