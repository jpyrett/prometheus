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
from pathlib import Path

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
BOB_SESSION = Path(__file__).parent / "fixtures" / "session_real" / "bob_jones_oil.json"


def oil_four(**options):
    """The real oil session decided again over four companies, evenly: EOG, CVX (approved) and CNQ, OXY (neutral): $3,750.00 each of the $15,000.00."""
    return account(profile=OIL_USER, fractional_even_split=options.pop("fractional_even_split", True)).decide_trades(oil_order(), OIL_CASH, None, MOMENT, "auto", "even", None, count=4)


def backtest(order, portfolio=OIL_CASH, profile=OIL_USER):
    account(profile=profile).perform_trades(order, portfolio, trader(mode="backtest"), EVENING)
    return {t["symbol"]: t for t in order.trades}


def test_the_real_opens_are_what_the_data_clerk_reads_from_the_stored_day_file():
    from handshake_helpers import clerk
    from datetime import date
    got = clerk().tools.daily_bars(list(OPENS), date(2026, 11, 13), date(2026, 11, 13))["bars"]
    assert {s: price(bars[0]["open"]) for s, bars in got.items()} == OPENS      # the stored bars are market data (floats): a price goes through the price door


def test_the_shipped_settings_have_no_slippage_so_a_paper_fill_is_at_the_quote_and_a_backtest_fill_at_the_open():
    from prometheus.agents.trader.simulator import fill_backtest, fill_paper
    slippage = load_config().costs.slippage_bps
    assert slippage == 0 and json.loads((Path(__file__).parent.parent / "configs" / "trader_v1.json").read_text(encoding="utf-8"))["costs"]["slippage_bps"] == 0
    paper = fill_paper("buy", "market", None, Decimal("170.59"), Decimal("168.00"), Decimal("172.00"), slippage)
    backtest = fill_backtest("buy", "market", None, {"date": "2026-11-13", "open": OPENS["XOM"], "high": 172.0, "low": 168.0, "close": 170.59}, Decimal("168.00"), Decimal("175.00"), slippage)      # a stored bar is market data: floats, and the open as the Data Clerk read it
    assert (paper.price, paper.slippage_bps, backtest.price, backtest.price_source) == (Decimal("170.59"), 0, Decimal("171.43"), "bar_open")
    assert fill_paper("buy", "market", None, Decimal("170.59"), Decimal("168.00"), Decimal("172.00"), 5).price == Decimal("170.6753")          # the simulator still takes the argument, and its own tests pass a number of their own


# ---------------------------------------------------------------- the decision

def test_every_purchase_of_an_even_split_is_an_order_in_dollars_with_its_estimate_and_the_pot_and_nothing_else_is(allowed):
    out = oil_four()
    got = {d["symbol"]: d for d in out.trade_decisions if d["will_trade"] == "yes_auto"}
    assert list(got) == ["EOG", "CVX", "CNQ", "OXY"] and {d["dollars"] for d in got.values()} == {Decimal("3750.00")} and {d["est_cost"] for d in got.values()} == {Decimal("3750.00")} and {d["size"]["pot"] for d in got.values()} == {Decimal("15000.00")}
    assert {s: d["price"] for s, d in got.items()} == {"EOG": Decimal("158.32"), "CVX": Decimal("219.89"), "CNQ": Decimal("49.75"), "OXY": Decimal("58.75")} and all(d["shares"] == sh.buys(Decimal("3750.00"), d["price"]) for d in got.values())      # the shares are the estimate at the sizing price
    assert [wo.verify_trade(out, d["record_id"], MOMENT) for d in got.values()] == [[]] * 4 and sum((d["dollars"] for d in got.values()), Decimal("0.00")) <= Decimal("15000.00")
    assert not [k for k in out.trade_decisions if k["will_trade"] != "yes_auto" and k.get("dollars")]
    robin = account(profile=OIL_USER).decide_trades(oil_order(), OIL_CASH, None, MOMENT, "auto", None, None, count=4)              # round robin (the default): whole shares, held by the pot
    assert [d for d in robin.trade_decisions if d.get("dollars")] == [] and {d["size"]["pot"] for d in robin.trade_decisions if d["size"]} == {Decimal("15000.00")} and all(type(d["shares"]) is Decimal and sh.whole(d["shares"]) for d in robin.trade_decisions)
    off = oil_four(fractional_even_split=False)                                                          # fractions off: nothing is an order in dollars, and M1 alone holds the pot
    assert [d for d in off.trade_decisions if d.get("dollars")] == [] and all(type(d["shares"]) is Decimal and sh.whole(d["shares"]) for d in off.trade_decisions) and {d["size"]["pot"] for d in off.trade_decisions if d["size"]} == {Decimal("15000.00")}


def test_an_even_split_held_down_by_the_shares_the_user_asked_for_keeps_that_purchase_an_order_in_shares(allowed):
    from prometheus.agents.account_manager import sizing as sz
    from test_fractional_even_split import purchases
    got = {s.symbol: s for s in sz.size_purchases(purchases("EOG", "CVX", "CNQ"), Decimal("30000.00"), allocation="even", fractional=True, wanted={"rec-EOG": 10})}
    assert got["EOG"].binding == "asked" and got["EOG"].dollars is None and {got["CVX"].dollars, got["CNQ"].dollars} == {Decimal("1583.20")} and all(s.pot == Decimal("30000.00") for s in got.values())


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


def test_traded_at_the_next_sessions_real_opens_each_fill_is_its_dollars_to_the_cent_more_shares_where_the_open_is_lower_fewer_where_it_is_higher(allowed):
    order = oil_four()
    estimate = {d["symbol"]: d["shares"] for d in order.trade_decisions if d["will_trade"] == "yes_auto"}
    t = backtest(order)
    assert {s: (x["status"], x["fill"]["price"], x["fill"]["price_source"], x["fill"]["date"]) for s, x in t.items()} == {s: ("filled", OPENS[s], "bar_open", "2026-11-13") for s in estimate}
    assert {s: x["fill"]["amount"] for s, x in t.items()} == {s: Decimal("3750.00") for s in estimate} and sum((x["fill"]["amount"] for x in t.values()), Decimal("0.00")) == Decimal("15000.00") and order.trader_runs[-1]["portfolio_after"]["cash"] == Decimal("0.00")
    assert {s: x["shares"] > estimate[s] for s, x in t.items()} == {"EOG": True, "CVX": True, "CNQ": True, "OXY": False}                              # the opens are below the sizing prices of the first three and above OXY's
    assert {s: x["fill"]["shares"] for s, x in t.items()} == {s: sh.buys(Decimal("3750.00"), OPENS[s]) for s in estimate} and all(x["dollars"] == Decimal("3750.00") and x["shares_approved"] == estimate[s] for s, x in t.items())
    assert all(f"an order of $3,750.00: {sh.text(x['shares'])} shares at {OPENS[s]:,.4f} (the estimate was {sh.text(estimate[s])})" in x["reason"] for s, x in t.items())
    assert wo.verify_trades(order) == [] and order.errors == [] and order.status == "traded"


def test_a_run_in_whole_shares_never_passes_its_pot_and_a_purchase_a_higher_open_would_push_past_it_is_cut_and_says_why():
    data = json.loads(BOB_SESSION.read_text(encoding="utf-8"))
    order = wo.WorkOrder.model_validate(copy.deepcopy(data["order"]))
    keep = {r["record_id"] for r in order.reviews if r["symbol"] == "XOM"}
    order.reviews = [r for r in order.reviews if r["record_id"] in keep]
    order.recommendations = [r for r in order.recommendations if r["identity"]["record_id"] in keep]
    order.status, order.trade_decisions, order.trades, order.trader_runs = "reviewed", [], [], []
    order = decisions.decide_trades(order, BOB, PROFILE, None, MOMENT, "auto", None, Decimal("1364.72"))                  # the pot is exactly eight XOM at the close: $1,364.72
    d = order.trade_decisions[0]
    assert (d["shares"], d["est_cost"], d.get("dollars"), d["size"]["pot"]) == (8, Decimal("1364.72"), None, Decimal("1364.72"))      # a round robin: an order in shares, held by the pot
    manager().perform_trades(order, BOB, trader(mode="backtest"), EVENING)
    t = order.trades[0]
    assert t["fill"]["price"] == OPENS["XOM"] and trade_amount(8, OPENS["XOM"]) > Decimal("1364.72")                      # eight at the real open would cost $1,371.44
    assert (t["status"], t["shares"], t["fill"]["amount"]) == ("filled", 7, Decimal("1200.01")) and "the size was cut from 8 to 7 shares for the money of this request (its pot)" in t["reason"]
    assert wo.verify_trades(order) == [] and t["fill"]["amount"] <= Decimal("1364.72")


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
