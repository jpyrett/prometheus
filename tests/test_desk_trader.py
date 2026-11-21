"""The desk with the Trader (Trader v1 spec sections 3 and 16): `build_desk(with_trader=True)` runs ... -> critic -> account_manager_decision -> [account_manager_funds] -> trader on ONE work order.

Everything is a fake: the Data Clerk marks the order ready, the Advisor puts the REAL captured PG_open recommendation on it, the Critic is the real one over a fake Data Clerk and a scripted model, the Trader is the real
one with no model and a ledger in memory, the Account Manager's Data Clerk answers the quotes from a dictionary, the account book is a fake store, and the clock is Thursday 2026-11-05 10:30 Eastern. Nothing here
calls a model, a quote, Postgres or a paid API.
"""
import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from advisor_helpers import MemoryLedger
from critic_helpers import NOW, portfolio, profile
from test_desk_critic import CapturedAdvisor, Manager, critic_for
from test_desk_graph import FakeClerk as DeskClerk, FakeResolver
from trader_helpers import MarketClerk, plan

from prometheus.agents.trader.trader import Trader
from prometheus.common.portfolio import Portfolio, apply_fill
from prometheus.langgraph import desk
from prometheus.persistence.accounts import Booked

REPO = Path(__file__).resolve().parents[1]
RUN_DESK = desk.run_desk          # a test of the command line replaces `desk.run_desk`; `run` below must still reach the real one
QUOTES = {"PG": 167.10, "MSFT": 420.0}
RICH = Portfolio(user="Jacob Pyrett", cash_avail=10000.0)      # the Account Manager sizes the purchase for this portfolio ($10,000 buys 59 shares of PG at 167.10); the cash pays for it, so no funding is needed


class FundsManager(Manager):
    """The real Account Manager with a RaiseFunds that answers with a given plan (and records what it was asked)."""

    def __init__(self, answer=None, **kw):
        super().__init__(**kw)
        self.answer = answer

    def raise_funds(self, portfolio_, target, shares, **kw):
        self.raised.append((target, shares, portfolio_.user))
        return self.answer or super().raise_funds(portfolio_, target, shares, **kw)


class BookStore:
    """The account book as the desk reaches it, without Postgres. `fail` is an exception raised by every call; `calls` records them."""

    def __init__(self, held=None, fail=None, seed="jacob_sample"):
        self.held, self.fail, self.seed = held or Portfolio(user="Jacob Pyrett", cash_avail=10000.0), fail, seed
        self.calls, self.booked, self.saved = [], [], []

    def _check(self, name, *args):
        self.calls.append((name, *args))
        if self.fail:
            raise self.fail

    def account_id_for_seed(self, key):
        self._check("seed", key)
        if key != self.seed:
            raise ValueError(f"no sample account {key!r}")
        return 7

    def list_portfolio(self, account_id):
        self._check("list_portfolio", account_id)
        return self.held

    def load_profile(self, account_id):
        self._check("load_profile", account_id)
        return profile(user=self.held.user)

    def get_user(self, account_id):
        self._check("get_user", account_id)
        return SimpleNamespace(display_name=self.held.user, status="active")

    def save_decisions(self, account_id, decisions):
        self._check("save_decisions", account_id)
        self.saved.extend(decisions)

    def book_trade(self, account_id, symbol, side, shares, price, traded_on, *, name=None, source="bot", record_id=None):
        """The real store's `book_trade` as far as the desk can tell (entry 82): the trade is applied to the held portfolio, so the account read again shows it, and a `Booked` comes back."""
        self._check("book_trade", account_id)
        before = self.held
        held_before = before.position(symbol).shares if before.position(symbol) else 0
        self.held = apply_fill(before, symbol, side, shares, price, traded_on)                                                                # the price is the Decimal the Account Manager booked at
        self.booked.append({"account_id": account_id, "symbol": symbol, "side": side, "shares": shares, "price": price, "traded_on": traded_on, "source": source, "record_id": record_id})
        return Booked(trade_row=len(self.booked), gain=Decimal("0.00"), symbol=symbol, side=side, shares=shares, price=price, cash_before=before.cash_avail,
                      cash_after=self.held.cash_avail, held_before=held_before, held_after=self.held.position(symbol).shares if self.held.position(symbol) else 0)

    def read_booked(self, account_id, trade_row):
        self._check("read_booked", account_id)
        return {**self.booked[trade_row - 1], "trade_id": trade_row} if 0 < trade_row <= len(self.booked) else None


def trader_with_ledger(**options):
    ledgers = []

    def factory(order):
        ledgers.append(MemoryLedger())
        return Trader(ledger=ledgers[-1], use_model=False, **options)

    factory.ledgers = ledgers
    return factory


def run(request=None, name="PG_open", approver=lambda card: True, manager=None, trader=None, funds_factory=None, **options):
    """The desk with the Critic and the Trader over fakes. `request` is what the user typed; the default is PG with the rich portfolio, so the purchase is paid with the cash. `manager` is the one Account Manager every
    node gets (it has Jacob's profile, so nothing is read from a file or a row); `funds_factory` makes a NEW one for each node, as the real desk does."""
    CapturedAdvisor.seen = None
    manager = manager or Manager(profile=profile(), clerk=MarketClerk(QUOTES))
    request = request or {"symbol": "PG", "from_days_ago": 14, "fetch": False, "account": {"portfolio": RICH}}
    return RUN_DESK(request, clerk_factory=lambda order: DeskClerk(), advisor_factory=lambda order: CapturedAdvisor(name), resolver=FakeResolver(), with_critic=True, critic_factory=critic_for(name),
                    funds_factory=funds_factory or (lambda: manager), approver=approver, clock=lambda: NOW, with_trader=True, trader_factory=trader or trader_with_ledger(), **options)


def edges(**options):
    g = desk.build_desk(clerk_factory=lambda order: None, advisor_factory=lambda order: None, **options).get_graph()
    return set(g.nodes), {(e.source, e.target, bool(e.conditional)) for e in g.edges}


END, START = "__end__", "__start__"


# ---------------------------------------------------------------- the graph

def test_a_trader_without_the_critic_is_refused_whatever_else_is_asked_for():
    for options in ({}, {"with_funds": True}, {"with_advisor": False}):
        with pytest.raises(ValueError, match="with_trader needs with_critic"):
            desk.build_desk(with_trader=True, **options)


def test_the_trader_node_follows_the_decision_and_the_graph_ends_after_it():
    nodes, found = edges(with_critic=True, with_trader=True)
    assert nodes == {START, "account_manager", "data_clerk", "advisor", "critic", "account_manager_decision", "account_manager_direct", "account_manager_funds", "trader", END}      # entry 142: the funds node is on every desk that decides AND trades
    assert found == {(START, "account_manager", False), ("account_manager", "data_clerk", True), ("account_manager", END, True), ("data_clerk", "advisor", True), ("data_clerk", END, True),
                     ("advisor", "critic", True), ("advisor", END, True), ("critic", "account_manager_decision", False),
                     ("account_manager_decision", "trader", True), ("account_manager_decision", END, True), ("trader", END, False),
                     ("account_manager", "account_manager_direct", True), ("account_manager_direct", "trader", True), ("account_manager_direct", END, True),
                     ("advisor", "account_manager_decision", True),      # the last four: a company the Advisor scored neutral counts as approved when nothing is left for the Critic (entry 91); and a trade the user names goes from the Account Manager's reading to its decision and on to the Trader (entry 87)
                     ("account_manager_decision", "account_manager_funds", True), ("account_manager_direct", "account_manager_funds", True), ("account_manager_funds", "trader", True), ("account_manager_funds", END, True)}      # entry 142: a purchase the user permitted to be paid for by selling holdings goes through RaiseFunds on its way to the Trader


def test_with_the_funds_node_the_funds_node_runs_first_and_its_answer_goes_on_to_the_trader():
    nodes, found = edges(with_critic=True, with_funds=True, with_trader=True)
    assert nodes == {START, "account_manager", "data_clerk", "advisor", "critic", "account_manager_decision", "account_manager_direct", "account_manager_funds", "trader", END}
    assert ("account_manager", "account_manager_direct", True) in found and ("account_manager_direct", "trader", True) in found and ("account_manager_direct", "account_manager_funds", True) in found      # a direct order the user permitted to be funded goes through the funds node too (entry 142; it was never funded under entry 87)
    assert ("account_manager_decision", "account_manager_funds", True) in found and ("account_manager_decision", END, True) in found
    assert ("account_manager_funds", "trader", True) in found and ("account_manager_funds", END, True) in found and ("trader", END, False) in found
    assert ("account_manager_decision", "trader", True) not in found and ("account_manager_funds", END, False) not in found       # the funds node does not end the graph by itself any more


def test_without_the_trader_the_graph_is_exactly_what_it_was_in_all_four_shapes():
    """The shapes of the desk before the Trader existed (Critic v1 and entry 42): the nodes and every edge, pinned (the Critic's desks also have the direct-order node of entry 87)."""
    base = {("account_manager", "data_clerk", True), ("account_manager", END, True), ("data_clerk", "advisor", True), ("data_clerk", END, True), (START, "account_manager", False)}
    assert edges() == ({START, "account_manager", "data_clerk", "advisor", END}, base | {("advisor", END, False)})
    assert edges(with_funds=True) == ({START, "account_manager", "data_clerk", "advisor", "account_manager_funds", END},
                                      base | {("advisor", "account_manager_funds", True), ("advisor", END, True), ("account_manager_funds", END, False)})
    critic = base | {("advisor", "critic", True), ("advisor", END, True), ("critic", "account_manager_decision", False), ("advisor", "account_manager_decision", True)}      # the last: a neutral company the user named counts as approved (entry 91)
    direct = {("account_manager", "account_manager_direct", True), ("account_manager_direct", END, True)}      # entry 87: the one thing the Critic's desks have since: a trade the user names goes from the Account Manager's reading to its direct decision (and, with the Trader, on to it)
    assert edges(with_critic=True) == ({START, "account_manager", "data_clerk", "advisor", "critic", "account_manager_decision", "account_manager_direct", END}, critic | direct | {("account_manager_decision", END, False)})
    assert edges(with_critic=True, with_funds=True) == ({START, "account_manager", "data_clerk", "advisor", "critic", "account_manager_decision", "account_manager_direct", "account_manager_funds", END},
                                                        critic | direct | {("account_manager_decision", "account_manager_funds", True), ("account_manager_decision", END, True), ("account_manager_funds", END, False)})


# ---------------------------------------------------------------- one order through the whole desk

def test_one_work_order_travels_to_the_trader_and_comes_out_traded_with_the_fill_in_memory():
    factory = trader_with_ledger()
    state = run(trader=factory)
    order = state["order"]
    assert state["path"] == ["account_manager", "data_clerk", "advisor", "critic", "account_manager_decision", "trader"] and state["errors"] == [] and order.errors == []
    assert order.status == "traded" and order.schema_version == 4 and [(t["symbol"], t["side"], t["status"], t["mode"], t["placed_by"], t["booked"]) for t in order.trades] == [("PG", "buy", "filled", "paper", "code", False)]
    assert order.trades[0]["shares"] == order.trade_decisions[0]["shares"] == 59 == Decimal("10000") // Decimal("167.1") and "approved_shares" not in order.reviews[0] and set(state["seconds"]) == {"account_manager", "data_clerk", "advisor", "critic", "account_manager_decision", "trader"}
    run_ = order.trader_runs[0]
    assert run_["use_model"] is False and run_["mode"] == "paper" and run_["booked"] is False and run_["portfolio_before"]["cash"] == Decimal("10000.0") and run_["portfolio_after"]["cash"] < Decimal("10000.0")
    assert [r["trade_id"] for r in factory.ledgers[0].records] == ["T1"]                                                     # the trades ledger of THIS Trader (a test never writes the real one)


def test_the_trader_is_given_the_clock_the_approver_and_the_order_the_decisions_were_made_on():
    seen = []

    class Watching(Trader):
        def execute_trades(self, order, account, portfolio=None, now=None, funding=None, approver=None, book=False):
            seen.append((order, now, funding, approver, book, portfolio.user, account.profile.user))
            return super().execute_trades(order, account, portfolio, now, funding, approver, book)

    ask = lambda card: True
    state = run(approver=ask, trader=lambda order: Watching(ledger=MemoryLedger(), use_model=False))
    [(order, now, funding, approver, book, user, holder)] = seen
    assert order is state["order"] and now == NOW and funding is None and approver is ask and book is False and user == holder == "Jacob Pyrett"


def test_the_trader_does_not_run_when_nothing_is_permitted_and_the_graph_ends_at_the_decision():
    calls = []
    trader = lambda order: calls.append(order) or Trader(ledger=MemoryLedger(), use_model=False)
    waiting = run(approver=None, trader=trader)                                                    # Ask mode and nobody to ask: pending_user, never a yes
    assert waiting["path"][-1] == "account_manager_decision" and waiting["order"].trade_decisions[0]["will_trade"] == "pending_user" and waiting["order"].trades == [] and waiting["order"].status == "reviewed"
    refused = run({"symbol": "SNDK", "fetch": False, "account": {"portfolio": RICH}}, name="SNDK", trader=trader)
    assert refused["order"].trade_decisions[0]["will_trade"] == "no_system_denied" and refused["path"][-1] == "account_manager_decision" and refused["order"].trades == []
    assert calls == [] and "TRADER" not in desk.report(waiting) and "TRADER" not in desk.report(refused)


def test_with_the_funds_node_a_graph_with_nothing_permitted_still_ends_after_the_funds_node_and_never_reaches_the_trader():
    manager = FundsManager(profile=profile(), clerk=MarketClerk(QUOTES))
    state = run({"symbol": "SNDK", "fetch": False, "funds": {"shares": 1}, "account": {"portfolio": RICH}}, name="SNDK", manager=manager, with_funds=True)
    assert state["path"][-2:] == ["account_manager_decision", "account_manager_funds"] and state["funds"]["status"] == "refused" and state["order"].trades == []


HOLDS = Portfolio(user="Jacob Pyrett", cash_avail=1000.0, positions=[{"symbol": "MSFT", "shares": 50, "cost_basis": 150.0, "acquired": "2025-01-02"}])      # $1,000 in cash and the sample's 50 shares of MSFT, in profit and old enough to sell


def test_the_funding_plan_of_the_funds_node_reaches_the_trader_which_does_not_sell_what_the_cash_already_covers():
    """Design choices entry 80: the Account Manager sizes a purchase by the cash that is there ($1,000 buys 5 shares of PG at 167.10; the 10% limit of the $22,000 equity would allow 13), so a purchase no longer NEEDS
    the funding plan. The plan of the funds node (it sells MSFT) still reaches the Trader; the Trader sees that the cash covers the purchase, sells nothing and asks the user no funding question: a sale is never
    made that is not needed, and never without a yes. (The funding question itself, and a declined one, are tested at the Trader: tests/test_trader.py.)"""
    asked = []
    manager = FundsManager(answer=plan("PG", ("MSFT", 20, 420.0)), profile=profile(), clerk=MarketClerk(QUOTES))
    state = run({"symbol": "PG", "fetch": False, "funds": {}, "account": {"portfolio": HOLDS}}, approver=lambda card: asked.append(card.get("kind", "decision")) or True, manager=manager, with_funds=True)
    order = state["order"]
    assert state["path"][-3:] == ["account_manager_decision", "account_manager_funds", "trader"] and manager.raised == [("PG", 5, "Jacob Pyrett")]      # the funds node was asked for the decision's own 5 shares
    assert asked == ["decision"]                                                                                           # only the decision's question: no sale is proposed
    assert [(t["trade_id"], t["kind"], t["side"], t["symbol"], t["shares"], t["status"]) for t in order.trades] == [("T1", "recommendation", "buy", "PG", 5, "filled")]
    assert order.status == "traded" and order.trader_runs[0]["funding"] == {"status": "plan", "approved": None, "note": "the cash covers the purchase: the funding sells are not used"}


# ---------------------------------------------------------------- the request: what the Trader is built from

def test_without_a_factory_the_node_builds_the_trader_from_the_trade_part_of_the_request(monkeypatch):
    built = []

    class Recording:
        def __init__(self, **kw):
            built.append(kw)

        def execute_trades(self, order, account, portfolio=None, now=None, funding=None, approver=None, book=False):
            built.append({"book": book})
            return order

    monkeypatch.setattr(desk, "Trader", Recording)
    for trade, expected in ((None, {"mode": None, "use_model": True}), ({"mode": "backtest", "book": True, "model": False}, {"mode": "backtest", "use_model": False}),
                            ({"mode": "paper", "model": True}, {"mode": "paper", "use_model": True})):
        built.clear()
        request = {"symbol": "PG", "fetch": False, "account": {"portfolio": RICH}, **({"trade": trade} if trade else {})}
        state = desk.run_desk(request, clerk_factory=lambda order: DeskClerk(), advisor_factory=lambda order: CapturedAdvisor(), resolver=FakeResolver(), with_critic=True, critic_factory=critic_for(),
                              funds_factory=lambda: Manager(profile=profile(), clerk=MarketClerk(QUOTES)), approver=lambda card: True, clock=lambda: NOW, with_trader=True)
        assert built == [expected, {"book": bool(trade and trade.get("book"))}] and state["path"][-1] == "trader"


def test_a_factory_a_test_gives_is_used_as_it_is_and_the_request_does_not_change_it():
    mine = Trader(ledger=MemoryLedger(), use_model=False, mode="backtest")
    state = run({"symbol": "PG", "fetch": False, "account": {"portfolio": RICH}, "trade": {"mode": "paper", "model": True}}, trader=lambda order: mine)
    assert state["order"].trader_runs[0]["mode"] == "backtest" and state["order"].trader_runs[0]["use_model"] is False


def test_a_trader_that_cannot_start_is_a_message_on_the_order_and_the_order_stays_decided():
    def broken(order):
        raise RuntimeError("the Trader's settings file is not valid")
    state = run(trader=broken)
    order = state["order"]
    assert order.status == "decided" and order.trades == [] and order.trader_runs == [] and "the Trader could not start: the Trader's settings file is not valid" in order.errors
    assert state["path"][-1] == "trader"


def test_a_bad_profile_file_is_a_message_on_the_order_at_the_trader_as_at_the_other_nodes():
    state = run({"symbol": "PG", "fetch": False, "account": {"portfolio": RICH}})
    assert state["order"].status == "traded"
    good = state["order"]
    bad = run({"symbol": "PG", "fetch": False, "account": {"profile": "/nonexistent/profile.json"}})["order"]
    assert any("the profile file does not exist" in e for e in bad.errors) and bad.trades == [] and good.errors == []


def test_the_book_flag_without_a_saved_account_keeps_the_fills_in_memory_and_says_so():
    state = run({"symbol": "PG", "fetch": False, "account": {"portfolio": RICH}, "trade": {"book": True}})
    order = state["order"]
    assert order.trades[0]["status"] == "filled" and order.trades[0]["booked"] is False and order.trader_runs[0]["booked"] is False
    assert any("the fills were not booked" in e and "no saved account" in e for e in order.trader_runs[0]["errors"])


def test_the_mode_backtest_runs_the_trader_against_the_stored_bars_not_the_quote():
    clerk = MarketClerk(QUOTES, bars={"PG": [], "SPY": []})
    state = run({"symbol": "PG", "fetch": False, "account": {"portfolio": RICH}}, manager=Manager(profile=profile(), clerk=clerk), trader=trader_with_ledger(mode="backtest"))
    order = state["order"]
    assert clerk.asked("quotes") == [] and clerk.asked("bars") == [["PG", "SPY"]] and order.trader_runs[0]["mode"] == "backtest"
    assert order.trades[0]["status"] == "active" and "waiting for the next session" in order.trades[0]["reason"] and order.status == "decided"   # no bar is stored from the next session on: the order waits


# ---------------------------------------------------------------- a saved account (--account)

def book_run(store, account="jacob_sample", trade=None, quotes=None, **kw):
    """The desk over a saved account: every node gets a NEW Account Manager with no profile of its own (as the real desk's default factory makes it), so the account's row is its profile unless the request gives one."""
    clerk = MarketClerk(QUOTES, **(quotes or {}))
    request = {"symbol": "PG", "fetch": False, "account": {"account": account}, **({"trade": trade} if trade else {})}
    return run(request, store_factory=lambda: store, funds_factory=lambda: Manager(clerk=clerk), **kw)


def test_a_saved_account_gives_the_profile_and_the_portfolio_and_the_run_is_connected_before_a_profile_is_set():
    store = BookStore()
    state = book_run(store)
    order = state["order"]
    assert state["errors"] == [] and order.status == "traded" and ("seed", "jacob_sample") in store.calls and ("list_portfolio", 7) in store.calls
    assert ("load_profile", 7) in store.calls                                                                               # the profile is the account's own row: nothing was given
    assert order.trader_runs[0]["portfolio_before"]["cash"] == Decimal("10000.0") and store.booked == []                                   # no --book: the fills stay in memory
    assert [d["will_trade"] for d in store.saved] == ["yes_user"]                                                           # a connected manager saves its decisions (entry 72): --account does it with or without --book


def test_the_report_says_that_a_saved_account_is_written_to_and_whether_the_fills_are_booked():
    text = desk.report(book_run(BookStore()), 1.0)
    assert "ACCOUNT  saved account jacob_sample (the Postgres account book): the Account Manager's decisions are saved in it; no fill is booked in it" in text
    booked = desk.report(book_run(BookStore(), trade={"mode": "paper", "book": True}), 1.0)
    assert "decisions are saved in it; the Trader's fills are booked in it" in booked
    assert "ACCOUNT  saved account" not in desk.report(run(), 1.0)                                                        # the JSON files are not an account book


def test_an_account_number_is_used_as_it_is_without_the_sample_lookup():
    store = BookStore()
    book_run(store, account="7")
    assert ("seed", "jacob_sample") not in store.calls and not any(c[0] == "seed" for c in store.calls) and ("list_portfolio", 7) in store.calls


def test_a_profile_that_was_given_is_set_after_connect_and_replaces_the_accounts_row():
    """`connect` forgets a profile read from the default file and keeps one that was given: the desk connects first and sets the profile after, so the given one wins and the row is never read."""
    store, clerk = BookStore(), MarketClerk(QUOTES)
    request = {"symbol": "PG", "fetch": False, "account": {"account": "jacob_sample", "profile": profile(trade_mode="auto")}}
    order = run(request, store_factory=lambda: store, funds_factory=lambda: Manager(clerk=clerk))["order"]
    assert not any(c[0] == "load_profile" for c in store.calls) and ("list_portfolio", 7) in store.calls
    assert order.trade_decisions[0]["mode"] == "auto" and book_run(BookStore())["order"].trade_decisions[0]["mode"] != "auto"                  # the given profile's mode, not the row's


def test_with_the_book_flag_the_fills_are_saved_in_the_account_and_the_trade_says_so():
    store = BookStore()
    state = book_run(store, trade={"mode": "paper", "book": True})
    order = state["order"]
    [booked] = store.booked
    assert (booked["account_id"], booked["symbol"], booked["side"], booked["source"], booked["record_id"]) == (7, "PG", "buy", "bot", order.trades[0]["record_id"]) and booked["shares"] == order.trades[0]["shares"]
    assert order.trades[0]["booked"] is True and order.trader_runs[0]["booked"] is True and order.errors == []


def test_a_backtest_never_books_even_when_asked_to():
    store = BookStore()
    state = book_run(store, trade={"mode": "backtest", "book": True}, quotes={"bars": {"PG": [], "SPY": []}}, trader=trader_with_ledger(mode="backtest"))
    assert store.booked == [] and any("never writes the account book" in e for e in state["order"].errors) and state["order"].trades == []


def test_a_database_failure_is_a_message_that_names_the_kind_and_never_the_connection_details():
    leak = "connection to server at 127.0.0.1, port 5432 failed: FATAL: password authentication failed for user prometheus_app (password: hunter2)"
    store = BookStore(fail=ConnectionError(leak))
    state = book_run(store)
    order = state["order"]
    said = " ".join(order.errors)
    assert "the Critic could not start: the account book could not be read (ConnectionError)" in said and order.reviews == [] and order.trades == []      # AMENDED (entry 105): the Advisor reads no account, so the Critic is the first node to report it
    for secret in ("127.0.0.1", "5432", "password", "hunter2", "prometheus_app"):
        assert secret not in said and secret not in json.dumps(order.model_dump(), default=str), secret


def test_each_node_reports_the_database_failure_as_it_reports_a_bad_profile_file():
    """The Critic is the first node that reads the account (the Advisor is given no portfolio, entry 105); with a good first read and a failing later one, each of the others reports it too."""
    class Dies(BookStore):
        def __init__(self, after, **kw):
            super().__init__(**kw)
            self.after, self.reads = after, 0

        def list_portfolio(self, account_id):
            self.reads += 1
            if self.reads > self.after:
                raise TimeoutError("could not connect to 10.0.0.9: password=secret")
            return super().list_portfolio(account_id)

    for after, node in ((0, "the Critic could not start"), (1, "the Account Manager could not decide"), (2, "the Trader could not start")):
        order = book_run(Dies(after))["order"]
        said = " ".join(order.errors)
        assert f"{node}: the account book could not be read (TimeoutError)" in said and "10.0.0.9" not in said and "secret" not in said, node


def test_a_missing_password_file_and_an_unknown_account_say_what_is_wrong_without_a_path_or_a_value():
    def no_file():
        raise FileNotFoundError("/home/someone/capstone/secrets/postgres_local.txt does not exist (run scripts/setup_postgres.sh)")
    said = " ".join(run({"symbol": "PG", "fetch": False, "account": {"account": "jacob_sample"}}, store_factory=no_file)["order"].errors)
    assert "the account book is not set up: the file with its password is missing (run scripts/setup_postgres.sh)" in said and "/home/someone" not in said
    unknown = " ".join(book_run(BookStore(), account="nobody")["order"].errors)
    assert "no sample account 'nobody'" in unknown


def test_the_default_store_is_the_local_one_and_the_desk_never_loads_the_postgres_driver_unless_it_is_used():
    code = ("import sys; import prometheus.langgraph.desk as d; before = [m for m in sys.modules if m.startswith(('psycopg', 'prometheus.persistence'))]; "
            "d.build_desk(clerk_factory=lambda o: None, advisor_factory=lambda o: None, with_critic=True, with_trader=True); "
            "after = [m for m in sys.modules if m.startswith(('psycopg', 'prometheus.persistence'))]; print(before, after)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin"}, timeout=120)
    assert out.returncode == 0 and out.stdout.strip() == "[] []", out.stderr
    assert desk.open_book.__module__ == "prometheus.langgraph.desk"


def test_the_funds_node_plans_for_the_saved_accounts_holdings_not_for_the_json_file():
    held = Portfolio(user="Jacob Pyrett", cash_avail=10.0, positions=[{"symbol": "KO", "shares": 30, "cost_basis": 50.0, "acquired": "2025-01-02"}])
    manager = FundsManager(profile=profile(), clerk=MarketClerk(QUOTES))
    seen = []
    manager.raise_funds = lambda portfolio_, target, shares, **kw: seen.append((portfolio_.symbols, portfolio_.cash_avail)) or {"status": "funded", "quotes": {}, "run": {}, "plan": None, "errors": [], "target": target, "shares": shares}
    store = BookStore(held=held)
    run({"symbol": "PG", "fetch": False, "funds": {"shares": 1, "buy": "PG"}, "account": {"account": "jacob_sample"}}, store_factory=lambda: store, manager=manager, with_funds=True)      # the user's own symbol: with $10 no purchase is permitted
    assert seen == [(["KO"], Decimal("10.0"))]


# ---------------------------------------------------------------- the report

def test_the_report_has_the_trader_section_after_the_critics_and_the_trader_in_the_timing_and_the_path_is_finished():
    state = run()
    text = desk.report(state, 1.0)
    assert text.index("CRITIC REVIEW  (model z-ai/glm-5.3") < text.index("TRADER  (no model: the program placed the orders; PAPER mode") and "T1" in text and "FILLED" in text and "account  cash $10,000.00 ->" in text
    assert "trader" in text.split("TIMING")[1] and "(stopped here)" not in text and text.splitlines()[next(i for i, l in enumerate(text.splitlines()) if l.startswith("PATH"))].endswith("account_manager_decision -> trader")


def test_a_decided_order_the_trader_did_not_reach_is_still_a_path_that_stopped_and_the_report_has_no_trader_section():
    state = run(approver=None)
    text = desk.report(state, 1.0)
    assert "(stopped here)" in text and "TRADER" not in text


def test_a_run_of_the_trader_that_leaves_a_resting_order_is_still_a_finished_path():
    state = run(trader=trader_with_ledger(mode="backtest"), manager=Manager(profile=profile(), clerk=MarketClerk(QUOTES, bars={"PG": [], "SPY": []})))
    text = desk.report(state)
    assert state["order"].status == "decided" and "(stopped here)" not in text and "ACTIVE" in text


# ---------------------------------------------------------------- the command line

class Captured:
    """`desk.run_desk` replaced by a function that records what `main` asked for and returns a state with an order of a given status."""

    def __init__(self, monkeypatch, tmp_path, status="traded", path=()):
        self.requests, self.options, self.status, self.path = [], [], status, list(path)
        monkeypatch.setattr(desk, "OUT_DIR", tmp_path / "out" / "data" / "processed")
        monkeypatch.setattr(desk, "run_desk", self)

    def __call__(self, request, **options):
        self.requests.append(request)
        self.options.append(options)
        order = desk.new_work_order("q", subject=["PG"])
        order.status = self.status
        return {"request": request, "order": order, "errors": [], "resolution": None, "funds": None, "path": self.path, "seconds": {}}


def test_trade_with_no_value_is_paper_and_the_trade_part_of_the_request_says_the_mode_the_book_and_the_model(monkeypatch, tmp_path):
    seen = Captured(monkeypatch, tmp_path)
    assert desk.main(["--symbol", "PG", "--critic", "--trade"]) == 0
    assert seen.requests[0]["trade"] == {"mode": "paper", "book": False, "model": True} and seen.options[0]["with_trader"] is True and seen.options[0]["with_critic"] is True
    assert desk.main(["--symbol", "PG", "--critic", "--trade", "backtest", "--no-trader-model"]) == 0
    assert seen.requests[1]["trade"] == {"mode": "backtest", "book": False, "model": False}
    assert desk.main(["--symbol", "PG", "--critic", "--trade", "paper", "--account", "jacob_sample", "--book"]) == 0
    assert seen.requests[2]["trade"] == {"mode": "paper", "book": True, "model": True}
    assert seen.requests[2]["account"] == {"account": "jacob_sample"}                                                      # the account's own profile and holdings: no JSON file is named
    desk.main(["--symbol", "PG", "--critic", "--account", "7", "--profile", "configs/profiles/jacob_pyrett.json", "--mode", "auto"])
    assert seen.requests[3]["account"] == {"account": "7", "profile": "configs/profiles/jacob_pyrett.json", "mode": "auto"} and "trade" not in seen.requests[3] and "with_trader" not in seen.options[3]


def test_without_trade_the_request_and_the_options_are_exactly_what_they_were(monkeypatch, tmp_path):
    seen = Captured(monkeypatch, tmp_path, status="decided")
    desk.main(["--symbol", "PG", "--critic"])
    assert seen.requests[0]["account"] == {"portfolio": str(desk.DEFAULT_PORTFOLIO), "profile": str(desk.DEFAULT_PROFILE)} and "trade" not in seen.requests[0]
    assert set(seen.options[0]) == {"with_advisor", "with_critic", "approver"}
    desk.main(["--symbol", "PG", "--critic", "--raise-funds", "--portfolio", "p.json", "--buy-shares", "3"])
    assert seen.requests[1]["funds"] == {"portfolio": "p.json", "shares": 3} and seen.requests[1]["account"]["portfolio"] == "p.json"


def test_with_a_saved_account_the_funds_node_is_not_given_a_portfolio_file(monkeypatch, tmp_path):
    seen = Captured(monkeypatch, tmp_path)
    desk.main(["--symbol", "PG", "--critic", "--raise-funds", "--account", "jacob_sample", "--buy-shares", "3", "--trade"])
    assert seen.requests[0]["funds"] == {"shares": 3} and seen.requests[0]["account"] == {"account": "jacob_sample"}


@pytest.mark.parametrize("args, message", [
    (["--symbol", "PG", "--trade"], "--trade needs --critic"),
    (["--symbol", "PG", "--critic", "--no-trader-model"], "go with --trade"),
    (["--symbol", "PG", "--critic", "--account", "jacob_sample", "--book"], "go with --trade"),
    (["--symbol", "PG", "--critic", "--trade", "--book"], "--book needs --account"),
    (["--symbol", "PG", "--critic", "--trade", "backtest", "--account", "jacob_sample", "--book"], "--book goes with --trade paper"),
    (["--symbol", "PG", "--account", "jacob_sample"], "--account goes with --critic"),
    (["--symbol", "PG", "--critic", "--trade", "live"], "invalid choice"),
    (["--show-graph", "--trade"], "--trade needs --critic"),
])
def test_the_flags_that_do_not_go_together_are_refused_by_the_parser_before_anything_is_paid_for(capsys, monkeypatch, tmp_path, args, message):
    Captured(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as stop:
        desk.main(args)
    assert stop.value.code == 2 and message in capsys.readouterr().err


def test_show_graph_shows_the_trader_node_and_with_raise_funds_the_funds_node_before_it(capsys):
    assert desk.main(["--show-graph", "--critic", "--trade"]) == 0
    plain = capsys.readouterr().out
    assert "trader(trader)" in plain and "account_manager_decision -.-> trader;" in plain and "trader --> __end__;" in plain and "account_manager_funds -.-> trader;" in plain      # entry 142: the funds node is on every desk that decides and trades (a decision that carries funding goes through it); the old "not in plain" held until then
    assert desk.main(["--show-graph", "--critic", "--raise-funds", "--trade"]) == 0
    funds = capsys.readouterr().out
    assert "trader(trader)" in funds and "account_manager_funds -.-> trader;" in funds and "account_manager_decision -.-> account_manager_funds;" in funds and "account_manager_decision -.-> trader;" not in funds
    assert desk.main(["--show-graph", "--critic"]) == 0
    assert "trader" not in capsys.readouterr().out


@pytest.mark.parametrize("status, code", [("traded", 0), ("decided", 0), ("reviewed", 0), ("recommended", 1), ("data_ready", 1)])
def test_the_exit_code_with_trade_is_0_for_an_order_that_is_traded_decided_or_reviewed(monkeypatch, tmp_path, status, code):
    Captured(monkeypatch, tmp_path, status=status)
    assert desk.main(["--symbol", "PG", "--critic", "--trade"]) == code


CRITIC_PATH = ["account_manager", "data_clerk", "advisor", "critic", "account_manager_decision"]
ADVISOR_PATH = ["account_manager", "data_clerk", "advisor"]


@pytest.mark.parametrize("status, path, code", [
    ("decided", CRITIC_PATH, 0),                  # the normal end of a Critic run: every verdict was turned into a decision
    ("reviewed", CRITIC_PATH, 0),                 # a decision waits for the user's answer: the run went as far as it can
    ("recommended", ADVISOR_PATH, 0),             # the Advisor recommended nothing, so the Critic was never asked: finished
    ("recommended", CRITIC_PATH, 1),              # the Critic ran and the order was never reviewed (it could not start, say): NOT finished
    ("data_ready", ["account_manager", "data_clerk"], 1),    # stopped at the Data Clerk although the Advisor was asked for
    ("data_failed", ["account_manager", "data_clerk"], 1),
])
def test_the_exit_code_of_a_critic_run_without_trade_is_0_when_the_path_reached_the_decision(monkeypatch, tmp_path, status, path, code):
    """A `--critic` run ends `decided` or `reviewed`, never `recommended`: it exited 1 and printed "(stopped here)" for every good run until this was fixed (2026-11-14)."""
    Captured(monkeypatch, tmp_path, status=status, path=path)
    assert desk.main(["--symbol", "PG", "--critic"]) == code


@pytest.mark.parametrize("status, code", [("recommended", 0), ("data_ready", 1), ("decided", 1)])
def test_the_exit_code_of_a_run_without_the_critic_is_what_it_was(monkeypatch, tmp_path, status, code):
    Captured(monkeypatch, tmp_path, status=status, path=ADVISOR_PATH)
    assert desk.main(["--symbol", "PG"]) == code


def test_the_exit_code_with_no_advisor_is_still_data_ready(monkeypatch, tmp_path):
    Captured(monkeypatch, tmp_path, status="data_ready", path=["account_manager", "data_clerk"])
    assert desk.main(["--symbol", "PG", "--no-advisor"]) == 0
    Captured(monkeypatch, tmp_path, status="recommended", path=ADVISOR_PATH)
    assert desk.main(["--symbol", "PG", "--no-advisor"]) == 1


def test_the_json_output_and_the_saved_file_carry_the_trades_and_the_runs(monkeypatch, tmp_path, capsys):
    out_dir = tmp_path / "out" / "data" / "processed"
    monkeypatch.setattr(desk, "OUT_DIR", out_dir)
    monkeypatch.setattr(desk, "run_desk", lambda request, **options: run({**request, "account": {"portfolio": RICH}}))
    assert desk.main(["--symbol", "PG", "--critic", "--trade", "--json", "--no-fetch"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["order"]["status"] == "traded" and data["order"]["trades"][0]["status"] == "filled" and data["order"]["trader_runs"][0]["mode"] == "paper" and data["request"]["trade"]["mode"] == "paper"
    saved = json.loads((out_dir / "desk_PG.json").read_text())
    assert saved["order"]["trades"] == data["order"]["trades"] and saved["request"]["trade"] == {"mode": "paper", "book": False, "model": True}
