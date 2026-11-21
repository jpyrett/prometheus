"""The Trader and the Account Manager shake hands on EVERY trade (design choices entry 81; design/trader_handshake_spec.md with its amendments of 2026-11-14).

Jacob's decision: the Trader places one trade at a time, tells the Account Manager when the order goes through, and only then gets the approval for the next one, so it cannot overdraw the funds; it holds no cash of its
own, the Account Manager and the Postgres book own the funds and the positions (ACID); no cash is reserved.

REAL DATA ONLY (`handshake_helpers.py`): the real Trader (`use_model=False`: the sweep places the orders, the free repeatable path), the real Account Manager, the real Data Clerk over the stored daily bars (the price of a
run is the stored close of Friday 2026-11-13), the real session's order (`tests/fixtures/session_real/bob_jones_oil.json`: real recommendations and reviews, XOM and EOG decided at 8 shares each) and the real sample
portfolio. The database tests run against the REAL local Postgres on THROWAWAY accounts they create and delete; the three sample accounts are read before and after every one of them and must not move. They are skipped when
Postgres is not reachable. The only wraps are `Recorder` (it records the real calls and calls through) and, in two tests, one fault injected into one real call (a rival booking, a lost connection) to make a real path run.
"""
import ast
import inspect
import json
import textwrap
import threading
import time
from contextlib import suppress
from datetime import date, datetime, timezone
from decimal import Decimal

import psycopg
import pytest
from handshake_helpers import BOB, FRIDAY, MOMENT, Recorder, close_of, friday_close, jacob_as_bob, manager, real_order, trader  # noqa: F401  (friday_close is a fixture)
from trader_helpers import plan

from prometheus.agents.trader.report import trade_section
from prometheus.common import work_order as wo
from prometheus.common.money import dumps, price_scale
from prometheus.common.portfolio import Portfolio, trade_amount
from prometheus.persistence.accounts import SECRETS, open_account_store
from prometheus.rag.config import REPO_ROOT

pytestmark = pytest.mark.usefixtures("friday_close")

YEAR = (date(2026, 1, 1), date(2026, 12, 31))
SAMPLES = ("jacob_sample", "avery_chen", "blake_okonkwo")


def by_id(order):
    return {t["trade_id"]: t for t in order.trades}


def last(order):
    return order.trader_runs[-1]


def amount(order, trade_id):
    return by_id(order)[trade_id]["fill"]["amount"]


# ---------------------------------------------------------------- the order of the handshake, with nothing booked (the Account Manager's working portfolio)

def test_the_order_of_calls_is_place_approve_record_for_every_trade_and_the_next_trade_starts_only_after_the_account_manager_answered(monkeypatch):
    order, am = real_order(), manager()
    rec = Recorder(monkeypatch, am)
    am.perform_trades(order, BOB, trader(), MOMENT)
    assert rec.kinds() == ["place T1", "approve T1", "record T1", "place T2", "approve T2", "record T2"]
    answers = {(kind, trade_id): answer for kind, trade_id, answer in rec.answers}
    assert answers["approve", "T1"].cash_avail == Decimal("15000.0") and answers["approve", "T1"].shares == 8
    assert answers["approve", "T2"].cash_avail == answers["record", "T1"].cash_avail == Decimal("15000.0") - amount(order, "T1")       # the second approval met the funds AFTER the first fill
    assert [t["status"] for t in order.trades] == ["filled", "filled"] and order.status == "traded" and order.errors == [] and wo.verify_trades(order) == []


def test_every_sale_is_booked_before_the_purchase_it_pays_for_is_approved_and_the_purchase_meets_the_cash_the_sale_brought(monkeypatch):
    """Funding (spec section 7): XOM is bought with the help of 4 MSFT of the REAL sample portfolio ($51.23 of cash). The sale is its own trade through the same steps; the Account Manager is only ASKED (a preview) whether
    the sale would pay for one share before anything is sold, and the purchase is approved only after the sale was recorded, against the cash the account then shows."""
    order, am = real_order(), manager()
    rec = Recorder(monkeypatch, am)
    funding = plan("XOM", ("MSFT", 4, close_of("MSFT")))
    am.perform_trades(order, jacob_as_bob(), trader(), MOMENT, funding=funding, approver=lambda card: True)
    kinds = rec.kinds()
    assert kinds == ["preview T1", "place T1", "preview T1", "approve T1-F1", "record T1-F1", "approve T1", "record T1", "place T2", "approve T2", "record T2"]      # the first preview is the consent step: does the cash cover it?
    assert kinds.index("record T1-F1") < kinds.index("approve T1")                                                      # booked before the purchase is approved
    answers = {(kind, trade_id): answer for kind, trade_id, answer in rec.answers}
    sale = by_id(order)["T1-F1"]["fill"]["amount"]
    assert answers["approve", "T1-F1"].cash_avail == Decimal("51.23") and answers["record", "T1-F1"].cash_avail == Decimal("51.23") + sale
    assert not [a for k, i, a in rec.answers if k == "preview"][0].approved and [a for k, i, a in rec.answers if k == "preview"][1].approved and answers["approve", "T1"].cash_avail == answers["record", "T1-F1"].cash_avail    # the purchase was approved against the cash the sale left
    t = by_id(order)
    assert [t[i]["status"] for i in ("T1", "T1-F1", "T2")] == ["filled"] * 3 and t["T1"]["shares"] == 8 and wo.verify_trades(order) == []
    assert last(order)["funding"]["approved"] is True and last(order)["portfolio_after"]["cash"] == answers["record", "T2"].cash_avail


def test_a_funded_purchase_the_sales_cannot_pay_for_asks_the_account_manager_first_and_sells_nothing(monkeypatch):
    """The pre-check that nothing is sold when the sales cannot pay for one share stays: the Account Manager answers the question (insufficient_cash), and no sale is approved or recorded."""
    order, am = real_order(), manager()
    rec = Recorder(monkeypatch, am)
    funding = plan("XOM", ("GM", 1, close_of("GM")))                                                                    # one share of GM brings nowhere near the price of a share of XOM
    am.perform_trades(order, jacob_as_bob(), trader(), MOMENT, funding=funding, approver=lambda card: True)
    t = by_id(order)
    assert (t["T1"]["status"], t["T1"]["reject_code"], t["T1-F1"]["status"], t["T1-F1"]["reject_code"]) == ("rejected", "insufficient_cash", "rejected", "funding_failed")
    assert "nothing was sold" in t["T1"]["reason"] and "approve T1-F1" not in rec.kinds() and "record T1-F1" not in rec.kinds()
    assert last(order)["portfolio_after"]["cash"] == Decimal("51.23") and wo.verify_trades(order) == []


def test_running_the_trader_twice_on_one_order_without_booking_does_not_spend_twice_and_does_not_place_a_filled_trade_again(monkeypatch):
    """Amendment 1: with nothing booked the fills of the first run live only on `order.trades`. The Account Manager applies them to its working portfolio before the first approval of the second run, so the second run
    starts from the cash the first one left, even though the caller hands it the same fresh portfolio again; and a `filled` trade is final (rule T3)."""
    order, am = real_order(), manager()
    am.perform_trades(order, BOB, trader(), MOMENT)
    trades_after_first = dumps(order.trades, sort_keys=True)
    spent = last(order)["portfolio_after"]["cash"]
    assert spent < Decimal("15000.0")
    rec = Recorder(monkeypatch, am)
    am.perform_trades(order, BOB, trader(), MOMENT)                                                                     # the SAME portfolio object again: $15,000 on paper, but the account was spent
    assert rec.events == [] and dumps(order.trades, sort_keys=True) == trades_after_first                           # nothing was placed, approved or recorded; the trades are untouched
    assert last(order)["portfolio_before"]["cash"] == spent == last(order)["portfolio_after"]["cash"] and len(order.trader_runs) == 2
    assert [p["symbol"] for p in last(order)["portfolio_after"]["positions"]] == ["XOM", "EOG"] and BOB.cash_avail == Decimal("15000.0")


def test_a_purchase_cut_for_cash_by_the_first_run_is_final_and_a_second_run_starts_from_the_cash_that_is_left():
    """Two purchases, $2,000: the first run buys XOM (8 shares) and EOG is cut to what the rest pays for. A second run on the same order, handed the same $2,000 again, may not buy the 4 missing shares of EOG: the cut
    trade is final (rule T3), it places nothing, and it starts from the cash that is left."""
    order, am = real_order(), manager()
    rich = BOB.model_copy(update={"cash_avail": Decimal("2000.0")})
    am.perform_trades(order, rich, trader(), MOMENT)
    t = by_id(order)
    assert (t["T1"]["shares"], t["T2"]["shares"]) == (8, 4) and "cut from 8 to 4 shares for the cash" in t["T2"]["reason"]
    left = last(order)["portfolio_after"]["cash"]
    assert 0 <= left < close_of("EOG")
    am.perform_trades(order, rich, trader(), MOMENT)
    assert last(order)["portfolio_before"]["cash"] == left == last(order)["portfolio_after"]["cash"] and by_id(order) == t


# ---------------------------------------------------------------- the single-trade output check (amendment 2): work_order.verify_fill, over the REAL trades of real runs

def a_real_run(funded=False):
    """The real Trader's run over the real order (paper, nothing booked): T1 XOM and T2 EOG filled; with `funded`, T1 is paid for by selling 4 MSFT of the real sample portfolio (T1-F1)."""
    order = real_order()
    if funded:
        manager().perform_trades(order, jacob_as_bob(), trader(), MOMENT, funding=plan("XOM", ("MSFT", 4, close_of("MSFT"))), approver=lambda card: True)
    else:
        manager().perform_trades(order, BOB, trader(), MOMENT)
    return order


def fill_check(order, trade_id, edit=None, others_edit=None):
    """verify_fill of one real trade of the order after `edit` changed that trade (a copy) and `others_edit` the order."""
    if others_edit:
        others_edit(order)
    trade = json.loads(dumps(wo.trade_of(order, trade_id)))
    if edit:
        edit(trade)
    return wo.verify_fill(order, trade, order.trades)


def test_the_real_fills_of_a_real_run_pass_the_single_trade_check_each_alone_and_among_the_others():
    order = a_real_run(funded=True)
    assert [t["trade_id"] for t in order.trades] == ["T1", "T1-F1", "T2"] and wo.verify_trades(order) == []
    for trade in order.trades:
        assert wo.verify_fill(order, trade, order.trades) == [], trade["trade_id"]                                       # the trade itself among `others` is left out
        assert (wo.verify_fill(order, trade, []) == []) is (trade["kind"] != "funding"), trade["trade_id"]                # alone, a purchase has nothing to clash with; a funding sale has no purchase on the order to name


@pytest.mark.parametrize("edit, message", [
    (lambda t: t["fill"].update(shares=9), "filled 9 shares but the decision gave 8"),
    (lambda t: t.update(symbol="MSFT"), "is for 'MSFT' but the decision is for 'XOM'"),
    (lambda t: t.update(side="sell"), "is a sell but the decision's action is 'open'"),
    (lambda t: t.update(record_id="nobody"), "recommendation nobody has no trade decision"),
    (lambda t: t.update(record_id=None), "recommendation None has no trade decision"),
    (lambda t: t.update(fill=None), "T1 is filled but has no fill"),
    # AMENDED (entry 110): the two rows said "whole number of shares", the reason now says "a number of shares above zero"; shares=2.5 became 0.1234567 (2.5 is a valid fill now, seven decimals is refused)
    (lambda t: t["fill"].update(price=0), "the fill needs a price and a number of shares above zero"),
    (lambda t: t["fill"].update(shares=0.1234567), "the fill needs a price and a number of shares above zero"),
    (lambda t: t.update(status="done"), "status 'done' is not one of new, active, filled, rejected"),
    (lambda t: t.update(trade_id=""), "has no trade_id"),
])
def test_each_way_one_fill_can_disagree_with_the_decisions_is_a_finding_before_it_is_reported(edit, message):
    findings = fill_check(a_real_run(), "T1", edit)
    assert any(message in f for f in findings), findings


def test_a_decision_that_changed_after_the_ticket_is_a_finding_and_so_is_a_second_filled_trade_for_one_recommendation():
    order = a_real_run()
    assert any("the decision 'no_user_denied'" in f for f in fill_check(order, "T1", others_edit=lambda o: o.trade_decisions[0].update(will_trade="no_user_denied")))
    order = a_real_run()
    t2 = json.loads(dumps(wo.trade_of(order, "T2")))
    t2.update(record_id=wo.trade_of(order, "T1")["record_id"], symbol="XOM")
    assert any("is a second filled trade for recommendation" in f and "(the first is T1)" in f for f in wo.verify_fill(order, t2, order.trades))
    denied = next(d["record_id"] for d in order.trade_decisions if d["will_trade"] == "no_system_denied")
    assert any("has the decision 'no_system_denied'" in f for f in fill_check(order, "T1", lambda t: t.update(record_id=denied)))


def test_a_funding_sale_is_held_to_its_purchase_being_on_the_order_and_to_being_a_sale_but_not_yet_to_the_purchase_being_filled():
    order = a_real_run(funded=True)
    assert any("T1-F1 funds 'T9', which is not a purchase on this order" in f for f in fill_check(order, "T1-F1", lambda t: t.update(funds_for="T9")))
    assert any("T1-F1 is a funding trade but is not a sale" in f for f in fill_check(order, "T1-F1", lambda t: t.update(side="buy")))
    sale, purchase = json.loads(dumps(wo.trade_of(order, "T1-F1"))), json.loads(dumps(wo.trade_of(order, "T1")))
    purchase.update(status="new", fill=None)
    assert wo.verify_fill(order, sale, [purchase]) == []                                                                # the sale fills first and its purchase is placed after it: nothing to say yet ...
    order.trades = [purchase if t["trade_id"] == "T1" else t for t in order.trades]
    assert any("T1-F1 is a filled funding sale but the purchase it pays for, T1, is new, not filled" in f for f in wo.verify_trades(order))      # ... and the whole-run guard names it at the end


def test_the_single_trade_check_changes_nothing_and_does_not_choke_on_damaged_entries():
    order = a_real_run()
    before = order.model_dump_json()
    trade = wo.trade_of(order, "T1")
    assert wo.verify_fill(order, trade, [*order.trades, "junk", None, 7]) == [] and order.model_dump_json() == before
    assert wo.verify_fill(order, "not a record", order.trades) == ["trade #1 is not a record"]


# ---------------------------------------------------------------- the Trader holds no cash (rule: the Account Manager is the only owner of the funds)

TRADER_FILES = sorted((REPO_ROOT / "prometheus" / "agents" / "trader").glob("*.py")) + [REPO_ROOT / "prometheus" / "tools" / "trader.py"]
FORBIDDEN_NAMES = {"apply_fill", "affordable_shares", "after_purchase", "_buy_into", "_sell_from", "_with_positions"}      # how a portfolio is changed, and how the cash is cut: the Account Manager's
DOORS = {"request_prices", "request_bars", "trading_portfolio", "apply_earlier_fills", "current_portfolio", "approve_trade", "preview_trade", "record_fill", "profile"}      # what the Trader asks of the Account Manager


def parents(tree):
    return {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


@pytest.mark.parametrize("path", TRADER_FILES, ids=lambda p: p.name)
def test_the_traders_code_holds_no_cash_and_changes_no_portfolio(path):
    """Read the source of every Trader module (the model's tools too): no `apply_fill` or any other way to change a portfolio or cut the cash, no `Portfolio` built, no arithmetic on `cash_avail`, no assignment to it,
    and no `portfolio` attribute assigned. The Trader may REPORT a figure the Account Manager gave it (it reads `.cash_avail` and `.positions` to say them) and may compare no amount with it (`is None` is not a comparison of money)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    up = parents(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES or isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
            raise AssertionError(f"{path.name} line {node.lineno}: {getattr(node, 'id', getattr(node, 'attr', ''))} is the Account Manager's, not the Trader's")
        if isinstance(node, ast.alias) and node.name in FORBIDDEN_NAMES:
            raise AssertionError(f"{path.name} imports {node.name}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Portfolio" and path.name != "replay.py":      # replay.py is WIRING: it hands the Account Manager the portfolio its `--cash` flag asks for
            raise AssertionError(f"{path.name} line {node.lineno}: the Trader builds no Portfolio")
        if isinstance(node, ast.Attribute) and node.attr in ("cash_avail", "portfolio"):
            parent = up.get(node)
            assert not isinstance(parent, (ast.BinOp, ast.AugAssign, ast.UnaryOp)), f"{path.name} line {node.lineno}: arithmetic on {node.attr}"
            assert not (isinstance(node.ctx, ast.Store)), f"{path.name} line {node.lineno}: {node.attr} is assigned"
            assert not (isinstance(parent, ast.Compare) and node.attr == "cash_avail" and not all(isinstance(op, (ast.Is, ast.IsNot)) for op in parent.ops)), f"{path.name} line {node.lineno}: the Trader compares no amount with the cash"
    from_portfolio = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module == "prometheus.common.portfolio" for a in n.names}
    assert from_portfolio <= {"Portfolio", "trade_amount"} | ({"load_portfolio"} if path.name == "replay.py" else set()), (path.name, from_portfolio)       # a type to say what it reads, and the rounding of a fill's own amount (replay.py, wiring, loads the file its flag names)


def test_the_broker_has_no_portfolio_and_the_only_cash_it_ever_reads_is_asked_of_the_account_manager():
    from trader_helpers import account, config, decided, make_broker
    broker, _ = make_broker(decided(), account(10000.0), quotes={"PG": 167.0}, cfg=config())
    assert not hasattr(broker, "portfolio") and not hasattr(broker, "cash") and broker.cash_now() == Decimal("10000.0")
    broker.account.trading_portfolio(account(777.0))
    assert broker.cash_now() == Decimal("777.0")                                                                                   # it asks every time: there is no figure of its own to go stale


def test_the_traders_modules_call_only_these_doors_of_the_account_manager():
    """Rule T5 after the handshake: everything the Trader does with the Account Manager is one of these calls (approve_trade and record_fill are two more doors). Not `notify_trades`, not `record_trade`, not the book."""
    used = set()
    for path in TRADER_FILES:
        if path.name == "replay.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute) and node.value.attr == "account":
                used.add(node.attr)                                                                                      # ctx.account.X
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "account":
                used.add(node.attr)                                                                                      # account.X
    assert used and used <= DOORS and {"approve_trade", "record_fill"} <= used, used
    assert not {"notify_trades", "record_trade", "list_portfolio", "store", "add_funds"} & used


# ---------------------------------------------------------------- the real Postgres: ACID, on throwaway accounts

@pytest.fixture(scope="module")
def store():
    if not SECRETS.is_file():
        pytest.skip("secrets/postgres_local.txt is not on this machine (run scripts/setup_postgres.sh)")
    try:
        opened = open_account_store()
        opened.ping()
    except Exception as error:
        pytest.skip(f"Postgres is configured but not reachable ({type(error).__name__})")
    return opened


class Book:
    """The real local Postgres for one test: throwaway accounts that are deleted at the end, and a check that the three sample accounts did not move."""

    def __init__(self, store):
        self.store, self.made = store, []
        self.before = self.samples()

    def samples(self):
        state = {}
        for seed in SAMPLES:
            account_id = self.store.account_id_for_seed(seed)
            state[seed] = (self.store.account_balance(account_id), self.store.list_portfolio(account_id).model_dump_json(), self.store.number_trades(account_id, *YEAR),
                           self.store.number_approved_recommendations(account_id, *YEAR), self.store.number_denied_recommendations(account_id, *YEAR))
        return state

    def account(self, cash: str, name: str = "Trader Holder") -> int:
        """A new throwaway account with this much cash (a string of dollars): deleted when the test ends."""
        account_id = self.store.create_account(name, "moderate")
        self.made.append(account_id)
        self.store.add_funds(account_id, Decimal(cash), date(2026, 11, 13))
        return account_id

    def manager(self, account_id: int):
        return manager(store=self.store, account_id=account_id)

    def spy(self, monkeypatch) -> list[str]:
        """Record, in order, every read of the funds (`list_portfolio`), every read of the account's status (`get_user`), every booking transaction (`book_trade`) and every read of a booked trade row (`read_booked`)
        the REAL store is asked for from now on; the call goes through."""
        log: list[str] = []
        for name in ("list_portfolio", "get_user", "book_trade", "read_booked"):
            real = getattr(self.store, name)

            def call(*args, _name=name, _real=real, **kw):
                log.append(_name)
                return _real(*args, **kw)

            monkeypatch.setattr(self.store, name, call)
        return log

    def cash(self, account_id: int) -> Decimal:
        return self.store.account_balance(account_id)

    def held(self, account_id: int) -> dict[str, Decimal]:
        return {p.symbol: p.shares for p in self.store.list_portfolio(account_id).positions}


@pytest.fixture
def db(store):
    book = Book(store)
    try:
        yield book
    finally:
        for account_id in book.made:
            with suppress(ValueError):
                store.delete_account(account_id)
        gone = [account_id for account_id in book.made if _exists(store, account_id)]
    assert gone == [], f"throwaway accounts left behind: {gone}"
    assert book.samples() == book.before, "a sample account moved"


def _exists(store, account_id) -> bool:
    try:
        store.get_user(account_id)
        return True
    except ValueError:
        return False


def test_two_purchases_that_cost_more_than_the_cash_the_first_is_booked_and_the_second_gets_what_the_book_has_left_and_the_cash_never_goes_below_zero(db, monkeypatch):
    account_id = db.account("2000.00")
    am, order = db.manager(account_id), real_order()
    rec = Recorder(monkeypatch, am)
    am.perform_trades(order, None, trader(), MOMENT, book=True)
    t = by_id(order)
    first, second = amount(order, "T1"), amount(order, "T2")
    assert (t["T1"]["status"], t["T1"]["shares"], t["T1"]["booked"], t["T2"]["status"], t["T2"]["shares"], t["T2"]["booked"]) == ("filled", 8, True, "filled", 4, True)
    assert "cut from 8 to 4 shares for the cash" in t["T2"]["reason"] and first + trade_amount(8, close_of("EOG")) > Decimal("2000")     # the decision's 8 EOG would have overdrawn it
    assert db.cash(account_id) == Decimal("2000.00") - first - second >= 0 and db.held(account_id) == {"XOM": 8, "EOG": 4}
    assert db.store.number_trades(account_id, *YEAR) == 2 and order.status == "traded" and order.errors == [] and last(order)["booked"] is True
    answers = {(kind, trade_id): answer for kind, trade_id, answer in rec.answers}
    assert answers["approve", "T2"].cash_avail == Decimal("2000.00") - first                                      # read from the BOOK after T1 was booked
    assert rec.kinds() == ["place T1", "approve T1", "record T1", "place T2", "approve T2", "record T2"] and [a.status for k, _, a in rec.answers if k == "record"] == ["booked", "booked"]


def test_a_second_purchase_the_remaining_cash_cannot_pay_for_even_one_share_of_is_rejected_insufficient_cash_and_the_book_keeps_the_rest(db):
    account_id = db.account("1400.00")
    am, order = db.manager(account_id), real_order()
    am.perform_trades(order, None, trader(), MOMENT, book=True)
    t = by_id(order)
    left = Decimal("1400.00") - amount(order, "T1")
    assert (t["T1"]["status"], t["T1"]["booked"], t["T2"]["status"], t["T2"]["reject_code"], t["T2"]["fill"], t["T2"]["booked"]) == ("filled", True, "rejected", "insufficient_cash", None, False)
    assert f"cash ${left:,.2f} does not pay for one share at" in t["T2"]["reason"] and db.cash(account_id) == left and db.held(account_id) == {"XOM": 8}
    assert db.store.number_trades(account_id, *YEAR) == 1 and order.status == "traded"


def test_after_a_run_the_book_and_the_runs_account_after_agree(db):
    account_id = db.account("2000.00")
    am, order = db.manager(account_id), real_order()
    am.perform_trades(order, None, trader(), MOMENT, book=True)
    saved, run = db.store.list_portfolio(account_id), last(order)
    assert run["portfolio_before"]["cash"] == Decimal("2000.0") and run["portfolio_after"]["cash"] == saved.cash_avail == db.cash(account_id)
    assert run["portfolio_after"]["positions"] == [{"symbol": p.symbol, "shares": p.shares, "cost_basis": p.cost_basis, "acquired": p.acquired.isoformat()} for p in saved.positions]
    assert [t["booked"] for t in order.trades] == [True, True] and run["booked"] is True and "booked in the account book" in "\n".join(trade_section(order))


def test_a_connected_account_with_book_off_is_traded_on_the_account_managers_working_portfolio_and_the_book_does_not_move(db, monkeypatch):
    account_id = db.account("15000.00")
    am, order = db.manager(account_id), real_order()
    rec = Recorder(monkeypatch, am)
    am.perform_trades(order, None, trader(), MOMENT, book=False)
    assert db.cash(account_id) == Decimal("15000.00") and db.held(account_id) == {} and db.store.number_trades(account_id, *YEAR) == 0      # the book was never written
    assert [a.status for k, _, a in rec.answers if k == "record"] == ["applied", "applied"] and [t["booked"] for t in order.trades] == [False, False] and last(order)["booked"] is False
    answers = {(kind, trade_id): answer for kind, trade_id, answer in rec.answers}
    assert answers["approve", "T2"].cash_avail == Decimal("15000.0") - amount(order, "T1") and last(order)["portfolio_after"]["cash"] < Decimal("15000.0")     # the second purchase still met the funds the first one left
    again = db.manager(account_id)
    again.perform_trades(order, None, trader(), MOMENT, book=True)                                                       # a later run that DOES book: the first run's fills were never in the book
    assert any("filled in an earlier run that did not book it, so it is not in the account book" in e for e in order.errors) and db.cash(account_id) == Decimal("15000.00") and db.held(account_id) == {}


def test_two_account_managers_racing_for_the_same_cash_the_book_accepts_one_purchase_and_refuses_the_other(db):
    """Two threads, each its OWN Account Manager over its own connections, on ONE account with $1,500: both read the cash, both are approved for the full 8 shares (XOM and EOG together cost about $2,600), both report
    their fill at the same moment. The book is the guard that cannot be raced: it locks the account row, so one booking is accepted and the other refused, and the cash is never below zero."""
    from test_account_manager_handshake import filled, tickets_of
    account_id = db.account("1500.00")
    order = real_order()
    tickets = tickets_of(order, Portfolio(user="Trader Holder", cash_avail=1500.0))
    prices = {"XOM": price_scale(close_of("XOM") * Decimal("1.0005")), "EOG": price_scale(close_of("EOG") * Decimal("1.0005"))}
    barrier, results, errors = threading.Barrier(2, timeout=30), {}, []

    def worker(symbol):
        try:
            am = db.manager(account_id)
            am.trading_portfolio(None, book=True)
            approval = am.approve_trade(tickets[symbol], prices[symbol], order=order, now=MOMENT, on=FRIDAY)
            assert approval.approved and approval.shares == 8 and approval.cash_avail == Decimal("1500.0"), approval          # both saw all the cash
            barrier.wait()                                                                                           # now both are approved against the same funds: only the book can stop the second
            results[symbol] = am.record_fill(filled(tickets[symbol], prices[symbol], approval.shares))
        except BaseException as error:   # a thread's failure must fail the test
            errors.append(error)
            barrier.abort()

    threads = [threading.Thread(target=worker, args=(s,)) for s in ("XOM", "EOG")]
    [t.start() for t in threads]
    [t.join(60) for t in threads]
    assert errors == [] and len(results) == 2
    assert sorted(r.status for r in results.values()) == ["booked", "refused"]
    won = next(s for s, r in results.items() if r.booked)
    lost = next(r for r in results.values() if not r.booked)
    cost = trade_amount(8, prices[won])
    assert lost.reason.startswith("cash $") and "does not cover" in lost.reason and lost.cash_avail is None
    assert db.cash(account_id) == Decimal("1500.00") - cost >= 0 and db.held(account_id) == {won: 8} and db.store.number_trades(account_id, *YEAR) == 1


def test_two_traders_racing_for_the_same_cash_one_purchase_is_booked_the_other_is_account_refused_and_nothing_overdraws(db):
    """The same race through the whole Trader: two Traders (each its own Account Manager and order) approve the same first purchase against the same $1,500, wait for each other, then both fill and report. One booking
    stands; the other trade is `rejected` `account_refused` with the book's sentence and no Fill; the next purchase of each then meets the cash the book has left (`insufficient_cash`)."""
    account_id = db.account("1500.00")
    barrier, outcome, errors = threading.Barrier(2, timeout=30), {}, []

    def run(name):
        try:
            am, order = db.manager(account_id), real_order()
            real, waited = am.approve_trade, []

            def approve_then_wait(trade, price, **kw):
                answer = real(trade, price, **kw)
                if not waited:
                    waited.append(1)
                    barrier.wait()                                                                                   # both have an approval for the first purchase, from the same cash
                return answer

            am.approve_trade = approve_then_wait
            am.perform_trades(order, None, trader(), MOMENT, book=True)
            outcome[name] = order
        except BaseException as error:
            errors.append(error)
            barrier.abort()

    threads = [threading.Thread(target=run, args=(n,)) for n in ("first", "second")]
    [t.start() for t in threads]
    [t.join(120) for t in threads]
    assert errors == [] and len(outcome) == 2
    t1 = [by_id(o)["T1"] for o in outcome.values()]
    assert sorted(t["status"] for t in t1) == ["filled", "rejected"]
    won, lost = next(t for t in t1 if t["status"] == "filled"), next(t for t in t1 if t["status"] == "rejected")
    assert (won["booked"], lost["reject_code"], lost["fill"], lost["booked"]) == (True, "account_refused", None, False) and lost["reason"].startswith("cash $") and "does not cover" in lost["reason"]
    assert {by_id(o)["T2"]["reject_code"] for o in outcome.values()} == {"insufficient_cash"}                            # EOG: the book had $134.60 left, not the price of a share
    cost = won["fill"]["amount"]
    assert db.cash(account_id) == Decimal("1500.00") - cost >= 0 and db.held(account_id) == {"XOM": 8} and db.store.number_trades(account_id, *YEAR) == 1


def test_a_booking_the_book_refuses_makes_the_trade_rejected_account_refused_and_the_next_approval_is_asked_only_after_that_answer(db, monkeypatch):
    """A rival (a second Account Manager on the same account) spends $1,700 between the first approval and the first booking: the book refuses T1's fill. T1 is `rejected` `account_refused` with the book's sentence and no
    Fill; T2 is approved only afterwards, against the cash the book really has."""
    account_id = db.account("2000.00")
    am, rival, order = db.manager(account_id), db.manager(account_id), real_order()
    rec = Recorder(monkeypatch, am)
    approve, done = am.approve_trade, []

    def approve_then_the_rival_spends(trade, price, **kw):
        answer = approve(trade, price, **kw)
        if trade["trade_id"] == "T1" and not done:
            done.append(1)
            rival.record_trade("XOM", "buy", 10, Decimal("170.00"), FRIDAY, source="user")                           # $1,700 of the $2,000
        return answer

    am.approve_trade = approve_then_the_rival_spends
    am.perform_trades(order, None, trader(), MOMENT, book=True)
    t = by_id(order)
    assert (t["T1"]["status"], t["T1"]["reject_code"], t["T1"]["fill"], t["T1"]["booked"]) == ("rejected", "account_refused", None, False)
    assert t["T1"]["reason"] == "cash $300.00 does not cover $1,364.72" and any(e == "T1: buy 8 XOM was not booked: cash $300.00 does not cover $1,364.72" for e in order.errors)      # AMENDED (entry 111): $1,365.40 was eight XOM at the close plus 5 bps; it is $1,364.72 with no slippage
    assert rec.kinds() == ["place T1", "approve T1", "record T1", "place T2", "approve T2", "record T2"]               # T2 was approved after T1's answer, not before it
    answers = {(kind, trade_id): answer for kind, trade_id, answer in rec.answers}
    assert answers["record", "T1"].status == "refused" and answers["approve", "T2"].cash_avail == Decimal("300.0") and (t["T2"]["status"], t["T2"]["shares"], t["T2"]["booked"]) == ("filled", 1, True)
    assert db.held(account_id) == {"XOM": 10, "EOG": 1} and db.cash(account_id) == Decimal("300.00") - amount(order, "T2") >= 0 and db.store.number_trades(account_id, *YEAR) == 2


def test_a_booking_lost_to_a_failed_connection_is_a_failed_trade_shown_to_the_user_never_retried_and_the_next_trade_is_placed_and_booked(db, monkeypatch):
    """Jacob, 2026-11-15: a failed trade is not committed to Postgres, the Account Manager marks it failed and shows the user the error, it is NOT retried, and the Trader goes on to the next trade. A lost connection
    (a real `psycopg.OperationalError` raised in place of the real `book_trade`'s FIRST call): the account is read again, it is unchanged, so nothing was committed. T1 is `rejected` `account_refused`, no connection
    detail is written anywhere, the user was shown the error and the account, and T2 is approved, placed and booked as if nothing had happened to T1."""
    account_id = db.account("2000.00")
    shown = []
    am, order = db.manager(account_id), real_order()
    am.screen = shown.append
    rec = Recorder(monkeypatch, am)
    real_book, calls = db.store.book_trade, []

    def lost_the_first_time(*args, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise psycopg.OperationalError("connection to server at 127.0.0.1 port 5432 failed: password authentication failed for user app")
        return real_book(*args, **kw)

    monkeypatch.setattr(db.store, "book_trade", lost_the_first_time)
    am.perform_trades(order, None, trader(), MOMENT, book=True)
    t = by_id(order)
    assert (t["T1"]["status"], t["T1"]["reject_code"], t["T1"]["fill"], t["T1"]["booked"]) == ("rejected", "account_refused", None, False)
    assert "the booking failed (OperationalError)" in t["T1"]["reason"] and "unchanged: nothing was committed" in t["T1"]["reason"]
    assert (t["T2"]["status"], t["T2"]["shares"], t["T2"]["booked"]) == ("filled", 8, True) and len(calls) == 2                      # T1 was tried once and never again; T2 was placed and booked
    assert rec.kinds() == ["place T1", "approve T1", "record T1", "place T2", "approve T2", "record T2"]
    text = "\n".join([json.dumps(order.model_dump(mode="json")), *shown])
    assert "127.0.0.1" not in text and "password" not in text and "5432" not in text
    said = "\n".join(shown)
    assert "TRADE NOT DONE: the buy of 8 XOM" in said and "will not be tried again" in said and "cash 2,000.00" in said      # the error, and the account as it stands: nothing moved
    assert order.status == "traded" and db.held(account_id) == {"EOG": 8} and db.store.number_trades(account_id, *YEAR) == 1


def test_an_account_that_is_not_active_is_not_approved_and_nothing_is_booked_or_placed(db):
    account_id = db.account("2000.00")
    db.store.update_user(account_id, status="suspended")
    am, order = db.manager(account_id), real_order()
    am.perform_trades(order, None, trader(), MOMENT, book=True)
    t = by_id(order)
    assert [(x["status"], x["reject_code"]) for x in order.trades] == [("rejected", "account_refused")] * 2 and "the account is suspended: it cannot take a trade" in t["T1"]["reason"]
    assert db.cash(account_id) == Decimal("2000.00") and db.held(account_id) == {} and db.store.number_trades(account_id, *YEAR) == 0


def test_the_book_is_read_for_every_approval_so_a_purchase_cannot_use_cash_the_book_no_longer_has(db):
    """`approve_trade` reads the cash from the book each time (not once at the start): a deposit and a withdrawal-by-purchase made by somebody else between two approvals change the answer."""
    from test_account_manager_handshake import tickets_of
    account_id = db.account("1000.00")
    am, rival, order = db.manager(account_id), db.manager(account_id), real_order()
    xom = tickets_of(order)["XOM"]
    am.trading_portfolio(None, book=True)
    ask = lambda: am.approve_trade(xom, Decimal("170.6753"), order=order, now=MOMENT, on=FRIDAY)
    assert (ask().shares, ask().cash_avail) == (5, Decimal("1000.0"))
    rival.add_funds(Decimal("500.00"), FRIDAY)
    assert (ask().shares, ask().cash_avail) == (8, Decimal("1500.0"))
    rival.record_trade("XOM", "buy", 8, Decimal("170.00"), FRIDAY, source="user")                                         # $1,360 of the $1,500
    answer = ask()
    assert (answer.approved, answer.shares, answer.code, answer.cash_avail) == (False, 0, "insufficient_cash", Decimal("140.0"))


# ---------------------------------------------------------------- the handshake never waits for a next trade (Jacob, 2026-11-14: "The handshaking should not wait if there are no further trades to execute")

OPEN, CLOSE = "list_portfolio", "list_portfolio"                      # the two reads of the book a run needs to report portfolio_before and portfolio_after
APPROVE = ["get_user", "list_portfolio"]                              # what one approval reads: the account's status, then the funds
RECORD = ["list_portfolio", "book_trade", "list_portfolio", "read_booked"]      # what one booking does (entry 82): SELECT the account, the ONE transaction, SELECT the account again, read the trade row back
LATER = datetime(2026, 11, 13, 21, 1, tzinfo=timezone.utc)            # a minute after the real approvals ran out


@pytest.fixture
def no_sleep(monkeypatch):
    """Any `time.sleep` during the test fails it: a handshake that waited for anything would be caught here (the quotes of these runs are the stored closes, so nothing else in a run sleeps)."""
    def refuse(seconds=0, *args):
        raise AssertionError(f"something called time.sleep({seconds}): the handshake must not wait")

    monkeypatch.setattr(time, "sleep", refuse)


def only_xom(order):
    """The real order after the user said no to EOG (a real state the Account Manager writes): one ticket is left."""
    for d in order.trade_decisions:
        if d["symbol"] == "EOG":
            d["will_trade"] = "no_user_denied"
    return order


def nothing_permitted(order):
    for d in order.trade_decisions:
        if d["will_trade"] in wo.TRADING:
            d["will_trade"] = "no_user_denied"
    return order


def handshake_calls(rec):
    return [kind for kind, _ in rec.events if kind in ("approve", "record")]


def test_for_n_trades_that_fill_there_are_n_approvals_and_n_bookings_and_not_one_call_after_the_last_answer(db, monkeypatch, no_sleep):
    """Real Postgres, real order: two purchases fill (T1 XOM, T2 EOG). The calls of the real Account Manager and the reads of the real store are recorded: exactly 2 approvals and 2 bookings, in order, the last event is
    the second booking, and after it the run does nothing but read the account back once to report it (and the booking's own read-back before it)."""
    account_id = db.account("15000.00")
    am, order = db.manager(account_id), real_order()
    rec, reads = Recorder(monkeypatch, am), db.spy(monkeypatch)
    am.perform_trades(order, None, trader(), MOMENT, book=True)
    assert [t["status"] for t in order.trades] == ["filled", "filled"] and handshake_calls(rec) == ["approve", "record", "approve", "record"]
    assert rec.kinds() == ["place T1", "approve T1", "record T1", "place T2", "approve T2", "record T2"] and rec.kinds()[-1] == "record T2"                         # nothing was asked after the last answer
    assert reads == [OPEN, *APPROVE, *RECORD, *APPROVE, *RECORD, CLOSE]                                                                                         # and the book was read for nothing else


def test_a_single_trade_that_fills_is_one_approval_and_one_booking_and_the_run_goes_straight_to_its_end(db, monkeypatch, no_sleep):
    account_id = db.account("15000.00")
    am, order = db.manager(account_id), only_xom(real_order())
    rec, reads = Recorder(monkeypatch, am), db.spy(monkeypatch)
    am.perform_trades(order, None, trader(), MOMENT, book=True)
    assert [t["trade_id"] for t in order.trades] == ["T1"] and rec.kinds() == ["place T1", "approve T1", "record T1"]
    assert reads == [OPEN, *APPROVE, *RECORD, CLOSE] and order.status == "traded" and last(order)["portfolio_after"]["cash"] == db.cash(account_id)


def test_a_trade_the_account_manager_refuses_has_no_booking_and_nothing_is_asked_after_its_answer(db, monkeypatch, no_sleep):
    """$1,400: T1 fills and is booked; T2 is asked for its approval and REFUSED (insufficient_cash). A refused approval is the end of that trade: no `record_fill`, and the run ends right after the refusal."""
    account_id = db.account("1400.00")
    am, order = db.manager(account_id), real_order()
    rec, reads = Recorder(monkeypatch, am), db.spy(monkeypatch)
    am.perform_trades(order, None, trader(), MOMENT, book=True)
    assert rec.kinds() == ["place T1", "approve T1", "record T1", "place T2", "approve T2"] and handshake_calls(rec) == ["approve", "record", "approve"]
    assert [a.approved for k, _, a in rec.answers if k == "approve"] == [True, False] and by_id(order)["T2"]["reject_code"] == "insufficient_cash"
    assert reads == [OPEN, *APPROVE, *RECORD, *APPROVE, CLOSE]                                                          # the refusal's own reads, then the one read that reports the account: nothing waits for a next trade


def test_the_same_holds_without_a_saved_account_on_the_account_managers_working_portfolio(monkeypatch, no_sleep):
    order, am = real_order(), manager()
    rec = Recorder(monkeypatch, am)
    am.perform_trades(order, BOB, trader(), MOMENT)
    assert handshake_calls(rec) == ["approve", "record", "approve", "record"] and rec.kinds()[-1] == "record T2" and [t["status"] for t in order.trades] == ["filled", "filled"]
    poor, order = manager(), real_order()
    rec = Recorder(monkeypatch, poor)
    poor.perform_trades(order, jacob_as_bob(), trader(), MOMENT)                                                          # the real sample's $51.23 pays for neither purchase
    assert rec.kinds() == ["place T1", "approve T1", "place T2", "approve T2"] and not [k for k in handshake_calls(rec) if k == "record"]
    assert [t["reject_code"] for t in order.trades] == ["insufficient_cash", "insufficient_cash"] and order.status == "traded"


def test_with_nothing_permitted_there_is_no_handshake_at_all_and_the_book_is_read_only_to_report_the_account(db, monkeypatch, no_sleep):
    account_id = db.account("15000.00")
    am, order = db.manager(account_id), nothing_permitted(real_order())
    rec, reads = Recorder(monkeypatch, am), db.spy(monkeypatch)
    am.perform_trades(order, None, trader(), MOMENT, book=True)
    assert order.trades == [] and rec.events == [] and reads == [OPEN, CLOSE]                                             # no approval, no booking, no place; one read for portfolio_before, one for portfolio_after
    assert last(order)["portfolio_before"]["cash"] == last(order)["portfolio_after"]["cash"] == Decimal("15000.0") and db.cash(account_id) == Decimal("15000.00")


def test_with_everything_already_final_from_an_earlier_run_there_is_no_handshake_and_no_further_read(db, monkeypatch, no_sleep):
    account_id = db.account("15000.00")
    order = real_order()
    db.manager(account_id).perform_trades(order, None, trader(), MOMENT, book=True)
    assert [t["booked"] for t in order.trades] == [True, True]
    again = db.manager(account_id)
    rec, reads = Recorder(monkeypatch, again), db.spy(monkeypatch)
    again.perform_trades(order, None, trader(), MOMENT, book=True)
    assert rec.events == [] and reads == [OPEN, CLOSE] and len(order.trader_runs) == 2 and db.store.number_trades(account_id, *YEAR) == 2


def test_with_every_ticket_refused_before_it_was_placed_there_is_no_handshake_at_all(db, monkeypatch, no_sleep):
    """The real approvals ran out a minute ago: both tickets are `rejected` `approval_expired` when they are built, so nothing is placed, approved or booked, and no price is even asked for."""
    account_id = db.account("15000.00")
    am, order = db.manager(account_id), real_order()
    rec, reads = Recorder(monkeypatch, am), db.spy(monkeypatch)
    am.perform_trades(order, None, trader(), LATER, book=True)
    assert [(t["status"], t["reject_code"]) for t in order.trades] == [("rejected", "approval_expired")] * 2 and rec.events == [] and reads == [OPEN, CLOSE]
    assert db.cash(account_id) == Decimal("15000.00") and db.held(account_id) == {}


def test_a_trade_with_no_price_is_rejected_before_it_fills_and_never_reaches_the_account_manager(db, monkeypatch, no_sleep):
    """The simulated Saturday has no stored bar, so the real Data Clerk has no quote: both tickets are `price_unavailable`. Nothing was filled, so nothing is approved or booked and nothing waits for a price."""
    monkeypatch.setattr("prometheus.agents.data_clerk.data_clerk_agent.market_day", lambda *a, **k: date(2026, 11, 14))
    account_id = db.account("15000.00")
    am, order = db.manager(account_id), real_order()
    rec, reads = Recorder(monkeypatch, am), db.spy(monkeypatch)
    am.perform_trades(order, None, trader(), MOMENT, book=True)
    assert [t["reject_code"] for t in order.trades] == ["price_unavailable"] * 2 and handshake_calls(rec) == [] and reads == [OPEN, CLOSE]


def test_a_resting_order_is_never_approved_or_booked_and_the_run_does_not_wait_for_the_market(monkeypatch, no_sleep):
    """A backtest at the moment of the decision: the next session's bar is not stored yet, so each order rests (`active`). A resting trade is not a trade about to be placed: no approval, no booking, no waiting."""
    order, am = real_order(), manager()
    stored = am.request_bars
    monkeypatch.setattr(am, "request_bars", lambda symbols, start, end: stored(symbols, min(start, FRIDAY), FRIDAY))      # the real stored bars AS THEY WERE at the moment of the decision: nothing after that Friday (the store has later sessions now, which would fill the orders)
    rec = Recorder(monkeypatch, am)
    am.perform_trades(order, BOB, trader(mode="backtest"), MOMENT)
    assert [t["status"] for t in order.trades] == ["active", "active"] and handshake_calls(rec) == [] and last(order)["portfolio_after"]["cash"] == Decimal("15000.0")


def test_with_the_funding_precheck_nothing_is_approved_for_a_purchase_that_would_not_fill_and_no_approval_is_left_pending(monkeypatch, no_sleep):
    """A funded purchase whose price rules do not fill it (the simulated Saturday has no quote): not asked, not sold, not booked: the question of the funding pre-check is only put for a purchase about to be placed."""
    monkeypatch.setattr("prometheus.agents.data_clerk.data_clerk_agent.market_day", lambda *a, **k: date(2026, 11, 14))
    order, am = real_order(), manager()
    rec = Recorder(monkeypatch, am)
    am.perform_trades(order, jacob_as_bob(), trader(), MOMENT, funding=plan("XOM", ("MSFT", 4, 523.88)), approver=lambda card: True)
    assert [t["reject_code"] for t in order.trades if t["kind"] == "recommendation"] == ["price_unavailable"] * 2 and rec.events == []


def test_the_traders_code_has_no_wait_of_any_kind_no_sleep_no_lock_no_thread_no_queue_no_poll_no_loop():
    """Rule of the handshake: nothing waits for a next trade. The source of the Trader's funnel, its steps, its class and the Account Manager's handshake module (and the methods of the Account Manager that the Trader
    calls) holds no import of a module that waits, no name that waits (`sleep`, `wait`, `poll`, an Event, a Condition, a Queue, a Thread, a Timer ...) and no `while` loop (a poll is one)."""
    from prometheus.agents.account_manager import account_manager, handshake
    from prometheus.agents.trader import broker, execute, trader
    waiting_modules = {"time", "threading", "asyncio", "queue", "sched", "select", "selectors", "multiprocessing", "concurrent", "signal"}
    waiting_names = {"sleep", "wait", "wait_for", "poll", "Event", "Condition", "Semaphore", "BoundedSemaphore", "Barrier", "Queue", "SimpleQueue", "Timer", "Thread", "Lock", "RLock"}
    sources = {module.__name__: textwrap.dedent(inspect.getsource(module)) for module in (broker, execute, trader, handshake)}
    for name in ("trading_portfolio", "current_portfolio", "apply_earlier_fills", "approve_trade", "preview_trade", "_approve", "record_fill"):
        sources[f"AccountManager.{name}"] = textwrap.dedent(inspect.getsource(getattr(account_manager.AccountManager, name)))
    for where, source in sources.items():
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                assert not {a.name.split(".")[0] for a in node.names} & waiting_modules, f"{where} imports a module that waits"
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in waiting_modules, f"{where} imports from a module that waits"
            elif isinstance(node, ast.Name):
                assert node.id not in waiting_names, f"{where} line {node.lineno}: {node.id}"
            elif isinstance(node, ast.Attribute):
                assert node.attr not in waiting_names, f"{where} line {node.lineno}: .{node.attr}"
            assert not isinstance(node, (ast.While, ast.AsyncFor, ast.AsyncWith, ast.Await)), f"{where} line {node.lineno}: a loop that could poll, or an await"
