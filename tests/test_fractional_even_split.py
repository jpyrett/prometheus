"""The EVEN split in fractional shares: "split $30,000 evenly over three stocks" buys $10,000.00 of each (design choices entry 110; design/fractional_shares_spec.md section 4).

REAL DATA ONLY: the prices are the real prices of the real sessions in `tests/fixtures/session_real/` (what each decision was sized at), and the real oil session (`auto_oil_no_second_mark.json`) is decided again by the real
Account Manager and traded by the real Trader (the stored closes of Friday 2026-11-13, `handshake_helpers.py`). The whole-share split of entry 108 (Part G) is the fallback and its own tests stay as they were: here is
the proof that the switch off gives exactly it. The connected-book tests use the real local Postgres on throwaway accounts (skipped when it is not reachable).
"""
import glob
import itertools
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest
from handshake_helpers import FRIDAY, MOMENT, close_of, friday_close, trader  # noqa: F401  (friday_close is a fixture)
from test_neutral_recommendation import OIL_CASH, OIL_USER, account, allowed, oil_order  # noqa: F401  (allowed is a fixture)
from test_trader_handshake import YEAR, db, store  # noqa: F401  (db and store are fixtures)

from prometheus.agents.account_manager import sizing as sz
from prometheus.agents.account_manager.config import load_config
from prometheus.common import shares as sh
from prometheus.common import work_order as wo
from prometheus.common.money import money, price
from prometheus.common.portfolio import trade_amount

pytestmark = pytest.mark.usefixtures("friday_close")

CAPTURES = Path(__file__).parent / "fixtures" / "session_real"


def real_prices() -> dict[str, Decimal]:
    """{symbol: the price a real decision was sized at}, from every real session capture."""
    found = {}
    for name in sorted(glob.glob(str(CAPTURES / "*.json"))):
        for d in (json.loads(Path(name).read_text(encoding="utf-8")).get("order") or {}).get("trade_decisions", []):
            if d.get("shares") and d.get("est_cost"):
                found[d["symbol"]] = price(d["price"]) if d.get("price") else price((money(d["est_cost"]) / sh.quantity(d["shares"])).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))      # the capture's floats go through the doors of money and shares
    return found


PRICES = real_prices()


def purchases(*symbols: str) -> list[sz.Purchase]:
    return [sz.Purchase(record_id=f"rec-{s}", symbol=s, action="open", price=PRICES[s], rank=n) for n, s in enumerate(symbols, 1)]


def size(symbols, pot, **options):
    return sz.size_purchases(purchases(*symbols), pot, allocation="even", fractional=True, **options)


# ---------------------------------------------------------------- the example of Jacob's words

def test_thirty_thousand_dollars_split_evenly_over_three_real_stocks_is_ten_thousand_dollars_of_each_and_the_cent_goes_to_the_first():
    got = size(["EOG", "CVX", "CNQ"], Decimal("30000.00"))
    assert [s.est_cost for s in got] == [Decimal("10000.00")] * 3 and [s.target_dollars for s in got] == [Decimal("10000.00")] * 3 and {s.of for s in got} == {3} and all(type(s.shares) is Decimal and not sh.whole(s.shares) for s in got)
    assert [s.shares for s in got] == [sh.buys(Decimal("10000.00"), PRICES[s.symbol]) for s in got] and [s.binding for s in got] == ["money"] * 3
    odd = size(["EOG", "CVX", "CNQ"], Decimal("30000.02"))                                                         # the remainder is divided again, one cent at a time, in rank order
    assert [s.est_cost for s in odd] == [Decimal("10000.01"), Decimal("10000.01"), Decimal("10000.00")] and {s.target_dollars for s in odd} == {Decimal("10000.01")} and sum(s.est_cost for s in odd) == Decimal("30000.02")
    assert [s.est_cost for s in size(["EOG", "CVX", "CNQ"], Decimal("30000.00"), budget=Decimal("9000.00"))] == [Decimal("3000.00")] * 3 and {s.binding for s in size(["EOG", "CVX", "CNQ"], Decimal("30000.00"), budget=Decimal("9000.00"))} == {"budget"}


def test_nobody_is_dropped_for_not_buying_one_whole_share_and_a_purchase_is_no_room_only_when_its_amount_buys_nothing():
    cheap_pot = size(["MU", "AMD", "F"], Decimal("600.00"))                                                          # $200 each: not one share of MU ($778.31) or AMD ($534.06), and the whole-share split drops them
    assert [s.est_cost for s in cheap_pot] == [Decimal("200.00")] * 3 and [s.shares for s in cheap_pot][:2] == [sh.buys(Decimal("200.00"), Decimal("778.31")), sh.buys(Decimal("200.00"), Decimal("534.06"))] and all(0 < s.shares for s in cheap_pot)
    whole = sz.size_purchases(purchases("MU", "AMD", "F"), Decimal("600.00"), allocation="even")
    assert [s.binding for s in whole][:2] == ["no_room", "no_room"]
    tiny = size(["EOG", "CVX", "CNQ"], Decimal("0.02"))
    assert [(s.est_cost, s.binding) for s in tiny] == [(Decimal("0.01"), "money"), (Decimal("0.01"), "money"), (Decimal("0.00"), "no_room")] and "buys nothing" in tiny[2].notes[0]


# ---------------------------------------------------------------- the three properties, on every real price

def test_every_amount_is_within_one_cent_of_the_others_the_total_is_not_above_the_pot_and_nothing_is_left_that_a_step_could_buy():
    for pot, k in itertools.product((Decimal("30000.02"), Decimal("15000.00"), Decimal("9999.99"), Decimal("2750.00"), Decimal("1234.57"), Decimal("100.00"), Decimal("1.00"), Decimal("0.07")), (1, 2, 3, 5)):
        for group in itertools.islice(itertools.combinations(sorted(PRICES), k), 0, None, 7):
            got = sz.size_purchases(purchases(*group), pot, allocation="even", fractional=True)
            paid = [s.est_cost for s in got if s.shares]
            total = sum((s.est_cost for s in got), Decimal(0))
            assert max(paid, default=Decimal(0)) - min(paid, default=Decimal(0)) <= Decimal("0.01"), (pot, group)
            assert total <= pot and all(s.est_cost == trade_amount(s.shares, PRICES[s.symbol]) for s in got)
            assert pot - total < min(PRICES[s] * sh.STEP for s in group) or total == pot, (pot, group)       # what is left is less than one step of any purchase (all prices are under $10,000: it is nothing)
            assert total == pot or not paid or len(paid) < k                                  # the pot is spent to the cent unless a purchase could buy nothing at all


def test_a_smaller_ask_holds_the_split_down_and_what_is_left_is_not_spent():
    got = {s.symbol: s for s in size(["EOG", "CVX", "CNQ"], Decimal("30000.00"), wanted={"rec-EOG": 10})}
    assert got["EOG"].shares == 10 and got["EOG"].est_cost == Decimal("1583.20") and got["EOG"].binding == "asked" and got["EOG"].asked == 10
    assert [got[s].est_cost for s in ("CVX", "CNQ")] == [Decimal("1583.20")] * 2 and {got[s].binding for s in ("CVX", "CNQ")} == {"even_split"} and {s.target_dollars for s in got.values()} == {Decimal("1583.20")}
    assert "what is left of the money is not spent" in got["CVX"].notes[1]
    roomy = {s.symbol: s for s in size(["EOG", "CVX", "CNQ"], Decimal("3000.00"), wanted={"rec-EOG": 10})}                  # an ask of more than the amount buys does not bind
    assert {s.est_cost for s in roomy.values()} == {Decimal("1000.00")} and roomy["EOG"].binding == "money"


def test_with_the_switch_off_or_not_asked_for_the_even_split_is_exactly_the_whole_share_split_and_round_robin_is_never_fractional():
    for symbols, pot in ((["EOG", "CVX", "CNQ"], Decimal("15000.00")), (["MU", "AMD", "F"], Decimal("5000.00")), (["MSFT", "XOM"], Decimal("2750.00"))):
        before = sz.size_purchases(purchases(*symbols), pot, allocation="even")
        off = sz.size_purchases(purchases(*symbols), pot, allocation="even", fractional=False)
        assert [s.model_dump() for s in before] == [s.model_dump() for s in off] and all(type(s.shares) is Decimal and sh.whole(s.shares) for s in off)
        for allocation in (None, "round_robin"):
            assert all(sh.whole(s.shares) for s in sz.size_purchases(purchases(*symbols), pot, allocation=allocation, fractional=True))
    assert [(s.shares, s.est_cost) for s in sz.size_purchases(purchases("EOG", "CVX", "CNQ"), Decimal("15000.00"), allocation="even")] == [(31, Decimal("4907.92")), (23, Decimal("5057.47")), (101, Decimal("5024.75"))]       # Part G, as it was pinned
    assert load_config().sizing.fractional_even_split is True


# ---------------------------------------------------------------- the real oil session, decided again and traded

def test_the_real_oil_session_decided_again_with_fractions_is_three_purchases_of_5000_dollars_and_nothing_is_rejected(allowed):
    told = []
    out = account(profile=OIL_USER, screen=told.append, fractional_even_split=True).decide_trades(oil_order(), OIL_CASH, None, MOMENT, "auto", "even", None, count=3)
    got = {d["symbol"]: d for d in out.trade_decisions}
    assert [(s, got[s]["will_trade"]) for s in ("EOG", "CVX", "CNQ")] == [("EOG", "yes_auto"), ("CVX", "yes_auto"), ("CNQ", "yes_auto")] and got["COP"]["will_trade"] == "no_system_denied" and out.errors == []
    assert [got[s]["est_cost"] for s in ("EOG", "CVX", "CNQ")] == [Decimal("5000.00")] * 3 and {got[s]["balance"]["target_dollars"] for s in ("EOG", "CVX", "CNQ")} == {Decimal("5000.00")} and not [d for d in out.trade_decisions if d["will_trade"] == "no_funds"]
    assert [got[s]["shares"] for s in ("EOG", "CVX", "CNQ")] == [Decimal("31.581606"), Decimal("22.738642"), Decimal("100.502512")] and "fractional shares" in got["EOG"]["size"]["notes"][0] and "all of it is spent" in got["EOG"]["size"]["notes"][1]
    assert sum(got[s]["est_cost"] for s in ("EOG", "CVX", "CNQ")) == OIL_CASH.cash_avail and not [line for line in told if "whole shares" in line]           # all of it spent, and the user is told nothing about whole shares
    assert [wo.verify_trade(out, got[s]["record_id"], MOMENT) for s in ("EOG", "CVX", "CNQ")] == [[], [], []]
    spent = account(profile=OIL_USER, fractional_even_split=False).decide_trades(oil_order(), OIL_CASH, None, MOMENT, "auto", "even", None, count=3)
    assert [(d["shares"], d["est_cost"]) for d in spent.trade_decisions if d["will_trade"] == "yes_auto"] == [(31, Decimal("4907.92")), (23, Decimal("5057.47")), (101, Decimal("5024.75"))]      # the switch off: Part G, $9.86 left


def test_the_three_fractional_purchases_are_traded_each_fills_its_dollars_to_the_cent_and_nothing_is_cut_or_rejected(allowed):
    """AMENDED (entry 111): with 5 bps of slippage the first two fills were $5,002.50 and the last was cut to the cash (CNQ 100.351783 for $4,995.00). There is no slippage now: each fill is its $5,000.00 and all of the $15,000.00 is spent."""
    order = account(profile=OIL_USER, fractional_even_split=True).decide_trades(oil_order(), OIL_CASH, None, MOMENT, "auto", "even", None, count=3)
    account(profile=OIL_USER).perform_trades(order, OIL_CASH, trader(), MOMENT)
    t = {x["symbol"]: x for x in order.trades}
    assert [(x["symbol"], x["status"]) for x in order.trades] == [("EOG", "filled"), ("CVX", "filled"), ("CNQ", "filled")] and wo.verify_trades(order) == [] and not [x for x in order.trades if x["status"] == "rejected"]
    assert all(type(x["fill"]["shares"]) is Decimal and not sh.whole(x["fill"]["shares"]) for x in order.trades) and order.trader_runs[-1]["portfolio_after"]["cash"] >= 0
    assert [x["fill"]["amount"] for x in order.trades] == [Decimal("5000.00")] * 3 and order.trader_runs[-1]["portfolio_after"]["cash"] == Decimal("0.00") and {x["symbol"]: x["fill"]["shares"] for x in order.trades} == {s: t[s]["shares_approved"] for s in t}


# ---------------------------------------------------------------- the account book decides whether a fraction may be bought

def test_an_account_book_that_cannot_hold_a_fraction_gets_the_whole_share_split_and_the_user_is_told_why(db, monkeypatch, allowed):
    account_id = db.account("15000.00", name=OIL_USER.user)
    told = []
    holds = db.store.holds_fractions()
    am = account(profile=OIL_USER, screen=told.append, fractional_even_split=None, store=db.store, account_id=account_id)
    if holds:                                                                                        # a migrated book: fractions, saved exactly
        out = am.decide_trades(oil_order(), OIL_CASH, None, MOMENT, "auto", "even", None, count=3)
        assert [d["shares"] for d in out.trade_decisions if d["will_trade"] == "yes_auto"] == [Decimal("31.581606"), Decimal("22.738642"), Decimal("100.502512")] and not [t for t in told if "whole shares" in t]
        assert db.store.number_approved_recommendations(account_id, *YEAR) == 3
    monkeypatch.setattr(db.store, "holds_fractions", lambda: False)                                   # the book as it was before `bash scripts/setup_postgres.sh`: the real method, answered the other way
    other = db.account("15000.00", name=OIL_USER.user)
    told.clear()
    old = account(profile=OIL_USER, screen=told.append, fractional_even_split=None, store=db.store, account_id=other)
    out = old.decide_trades(oil_order(), OIL_CASH, None, MOMENT, "auto", "even", None, count=3)
    assert [(d["shares"], d["est_cost"]) for d in out.trade_decisions if d["will_trade"] == "yes_auto"] == [(31, Decimal("4907.92")), (23, Decimal("5057.47")), (101, Decimal("5024.75"))]
    assert len([t for t in told if "the split is in whole shares" in t and "setup_postgres.sh" in t]) == 1 and out.errors == []
