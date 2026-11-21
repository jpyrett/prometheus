"""The rules of a waiting limit order that need no database (design choices entry 140; design/limit_orders_spec.md W1 to W3, W8): which order waits, which session a day order is for and when it ends (the market calendar's REAL sessions), what it
sets aside, what the user is told, and the ONE DOOR to what is available (a scan of the source, in the way `test_no_float_in_the_account.py` scans for floats).
"""
import ast
import inspect
import re
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from prometheus.agents.account_manager import orders
from prometheus.agents.account_manager.config import load_config
from prometheus.common.portfolio import Portfolio
from prometheus.persistence import accounts as accounts_module

ROOT = Path(__file__).resolve().parents[1]
NEW_YORK = orders.ZoneInfo("America/New_York")


def at(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=NEW_YORK)


# ---------------------------------------------------------------- W2: a day order, the session it is for and when it ends

@pytest.mark.parametrize("placed, session, ends_utc", [
    (at(date(2026, 11, 16), 10, 30), date(2026, 11, 16), datetime(2026, 11, 16, 21, 0, tzinfo=timezone.utc)),      # Monday, in the session: today's close, 16:00 EST
    (at(date(2026, 11, 16), 8, 0), date(2026, 11, 16), datetime(2026, 11, 16, 21, 0, tzinfo=timezone.utc)),       # Monday before the open: the session that is about to start
    (at(date(2026, 11, 13), 17, 0), date(2026, 11, 16), datetime(2026, 11, 16, 21, 0, tzinfo=timezone.utc)),      # Friday after the close: the next trading day
    (at(date(2026, 11, 14), 11, 0), date(2026, 11, 16), datetime(2026, 11, 16, 21, 0, tzinfo=timezone.utc)),      # Saturday: Monday's session
    (at(date(2026, 11, 26), 11, 0), date(2026, 11, 27), datetime(2026, 11, 27, 18, 0, tzinfo=timezone.utc)),      # Thanksgiving is a holiday: the next day is an EARLY CLOSE (13:00 Eastern)
    (at(date(2026, 11, 27), 12, 59), date(2026, 11, 27), datetime(2026, 11, 27, 18, 0, tzinfo=timezone.utc)),     # a minute before an early close: still that day's
    (at(date(2026, 11, 27), 13, 0), date(2026, 11, 30), datetime(2026, 11, 30, 21, 0, tzinfo=timezone.utc)),      # at the early close the session is over: the next Monday
])
def test_a_day_order_is_for_the_session_it_is_placed_in_or_the_next_one_and_ends_at_that_sessions_close(placed, session, ends_utc):
    assert orders.session_for(placed) == (session, ends_utc)


def test_a_moment_without_a_timezone_is_a_programming_mistake():
    with pytest.raises(ValueError, match="timezone"):
        orders.session_for(datetime(2026, 11, 16, 10, 30))


def test_the_user_is_told_the_exact_time_it_waits_until():
    monday_close = datetime(2026, 11, 16, 21, 0, tzinfo=timezone.utc)
    assert orders.until_text(monday_close, at(date(2026, 11, 16), 10, 30)) == "4:00 pm ET today"
    assert orders.until_text(monday_close, at(date(2026, 11, 13), 17, 0)) == "4:00 pm ET Monday 2026-11-16"
    assert orders.until_text(datetime(2026, 11, 27, 18, 0, tzinfo=timezone.utc), at(date(2026, 11, 26), 11, 0)) == "1:00 pm ET Friday 2026-11-27"


def test_the_watcher_looks_only_in_the_regular_session():
    assert orders.in_regular_session(at(date(2026, 11, 16), 9, 30)) and orders.in_regular_session(at(date(2026, 11, 16), 15, 59))
    assert not orders.in_regular_session(at(date(2026, 11, 16), 9, 29)) and not orders.in_regular_session(at(date(2026, 11, 16), 16, 0))      # the pre-market and the after-hours are not for filling
    assert not orders.in_regular_session(at(date(2026, 11, 14), 11, 0)) and not orders.in_regular_session(at(date(2026, 11, 26), 11, 0))      # a weekend, a holiday


# ---------------------------------------------------------------- W1: which orders wait

def entry(side="buy", symbol="MU", shares=None, dollars=None, max_price=None, min_price=None, how_many=None) -> dict:
    return {"side": side, "symbol": symbol, "shares": shares, "dollars": dollars, "max_price": max_price, "min_price": min_price, "how_many": how_many if how_many else ("typed" if shares is not None else None)}


@pytest.mark.parametrize("order, price, waits", [
    (entry("buy", shares=10, max_price=Decimal("90")), Decimal("95.12"), True),       # above the ceiling: waits
    (entry("buy", shares=10, max_price=Decimal("90")), Decimal("90.00"), False),      # at the ceiling: filled at once
    (entry("buy", shares=10, max_price=Decimal("90")), Decimal("89.99"), False),
    (entry("sell", shares=10, min_price=Decimal("600")), Decimal("590.10"), True),    # under the floor: waits
    (entry("sell", shares=10, min_price=Decimal("600")), Decimal("600.00"), False),
    (entry("buy", shares=10, min_price=Decimal("20")), Decimal("19.00"), False),      # a BUY under a floor does not wait (a stop order: not built)
    (entry("sell", shares=10, max_price=Decimal("90")), Decimal("95.12"), False),     # a sale is not limited by a ceiling
    (entry("buy", shares=10), Decimal("95.12"), False),                               # no limit: a market order, as ever
])
def test_only_a_purchase_above_its_ceiling_or_a_sale_under_its_floor_waits(order, price, waits):
    assert orders.waits(order, price) is waits


def test_a_limit_is_met_when_the_quote_is_at_or_inside_it():
    assert orders.meets("buy", Decimal("90.00"), Decimal("90.0000")) and not orders.meets("buy", Decimal("90.01"), Decimal("90"))
    assert orders.meets("sell", Decimal("600"), Decimal("600")) and not orders.meets("sell", Decimal("599.99"), Decimal("600"))


# ---------------------------------------------------------------- W3: what it sets aside, and what the user is told

def holder(cash="10000.00", **extra) -> Portfolio:
    return Portfolio(user="Plan Holder", cash_avail=Decimal(cash), positions=[{"symbol": "MSFT", "shares": 20, "cost_basis": Decimal("400.00"), "acquired": date(2026, 10, 1)}], **extra)


def test_a_purchase_sets_aside_its_most_possible_cost_and_the_user_is_told_in_the_words_of_the_spec():
    now = at(date(2026, 11, 20), 10, 0)      # Friday: this session
    plan = orders.plan_waiting(entry("buy", "MU", shares=Decimal(10), max_price=Decimal("90")), Decimal("95.12"), holder(), now, fractions=True)
    assert isinstance(plan, orders.Plan) and (plan.reserve_cash, plan.reserve_shares, plan.estimate, plan.session) == (Decimal("900.00"), Decimal(0), Decimal(10), date(2026, 11, 20))
    assert orders.waiting_sentence(plan, now) == "Waiting: buy 10 MU at $90.00 or less (now $95.12), $900.00 set aside, until 4:00 pm ET today."


def test_an_order_in_dollars_sets_aside_its_dollars_and_a_sale_its_shares():
    now = at(date(2026, 11, 20), 18, 0)      # after the close: Monday 2026-11-23
    dollars = orders.plan_waiting(entry("buy", "MU", dollars=Decimal("5000"), max_price=Decimal("90")), Decimal("95.12"), holder(), now, fractions=True)
    assert (dollars.reserve_cash, dollars.estimate) == (Decimal("5000"), Decimal("55.555555")) and dollars.session == date(2026, 11, 23)      # 5000 / 90 at six places
    assert orders.waiting_sentence(dollars, now) == "Waiting: buy $5,000.00 of MU at $90.00 or less (now $95.12), $5,000.00 set aside, until 4:00 pm ET Monday 2026-11-23."
    whole = orders.plan_waiting(entry("buy", "MU", dollars=Decimal("5000"), max_price=Decimal("90")), Decimal("95.12"), holder(), now, fractions=False)
    assert whole.estimate == Decimal(55)      # a book that holds whole shares only
    sale = orders.plan_waiting(entry("sell", "MSFT", shares=Decimal(5), min_price=Decimal("600")), Decimal("590.10"), holder(), now, fractions=True)
    assert (sale.reserve_cash, sale.reserve_shares) == (Decimal("0.00"), Decimal(5))
    assert orders.waiting_sentence(sale, now) == "Waiting: sell 5 MSFT at $600.00 or more (now $590.10), 5 shares set aside, until 4:00 pm ET Monday 2026-11-23."


@pytest.mark.parametrize("order, portfolio, will_trade, words", [
    (entry("buy", "MU", shares=Decimal(200), max_price=Decimal("90")), holder(), "no_funds", "sets aside the most it can cost: 200 shares at your limit of $90.00 is $18,000.00, and the cash available is $10,000.00"),
    (entry("buy", "MU", shares=Decimal(12), max_price=Decimal("90")), holder("1000.00", reserved_cash=Decimal("4000.00")), "no_funds", "$4,000.00 of it already set aside for waiting orders"),
    (entry("buy", "MU", max_price=Decimal("90")), holder(), "no_system_denied", "an order that waits needs an amount to set aside"),
    (entry("sell", "MSFT", shares=Decimal(25), min_price=Decimal("600")), holder(), "no_funds", "the account holds 20, so 20 can be set aside"),
    (entry("sell", "MSFT", shares=Decimal(15), min_price=Decimal("600")), holder(reserved_shares={"MSFT": Decimal(10)}), "no_funds", "10 of them already set aside for waiting orders, so 10 can be set aside"),
    (entry("sell", "MSFT", min_price=Decimal("600")), holder(), "no_system_denied", "no number of shares"),
    (entry("sell", "TSLA", shares=Decimal(1), min_price=Decimal("600")), holder(), "no_funds", "the account holds no TSLA"),
])
def test_what_cannot_set_its_cash_or_shares_aside_is_refused_with_the_reason(order, portfolio, will_trade, words):
    refusal = orders.plan_waiting(order, Decimal("95.12"), portfolio, at(date(2026, 11, 20), 10, 0), fractions=True)
    assert isinstance(refusal, orders.Refusal) and refusal.will_trade == will_trade and words in refusal.reason


def test_on_a_book_that_holds_whole_shares_only_dollars_that_buy_no_whole_share_at_the_limit_cannot_wait():
    now = at(date(2026, 11, 20), 10, 0)
    refusal = orders.plan_waiting(entry("buy", "MU", dollars=Decimal("50"), max_price=Decimal("90")), Decimal("95.12"), holder(), now, fractions=False)
    assert isinstance(refusal, orders.Refusal) and refusal.will_trade == "no_funds" and "does not buy even one share of MU at your limit of $90.00" in refusal.reason
    assert isinstance(orders.plan_waiting(entry("buy", "MU", dollars=Decimal("50"), max_price=Decimal("90")), Decimal("95.12"), holder(), now, fractions=True), orders.Plan)      # with fractions it buys a fraction of a share
    fraction = orders.plan_waiting(entry("buy", "MU", shares=Decimal("1.5"), max_price=Decimal("90")), Decimal("95.12"), holder(), now, fractions=False)
    assert isinstance(fraction, orders.Refusal) and "holds whole shares only" in fraction.reason


def test_a_sale_of_shares_bought_today_cannot_wait_for_a_session_that_is_still_today():
    """T+1: a share cannot be sold the day it was bought. A position bought on the session the order waits for has nothing to sell in it; one bought a day before is sellable."""
    today = date(2026, 11, 20)
    bought_today = Portfolio(user="Plan Holder", cash_avail=Decimal("0.00"), positions=[{"symbol": "MSFT", "shares": 20, "cost_basis": Decimal("400.00"), "acquired": today}])
    refusal = orders.plan_waiting(entry("sell", "MSFT", shares=Decimal(5), min_price=Decimal("600")), Decimal("590.10"), bought_today, at(today, 10, 0), fractions=True)
    assert isinstance(refusal, orders.Refusal) and "a share cannot be sold the day it was bought" in refusal.reason
    assert isinstance(orders.plan_waiting(entry("sell", "MSFT", shares=Decimal(5), min_price=Decimal("600")), Decimal("590.10"), bought_today, at(today, 18, 0), fractions=True), orders.Plan)      # placed after the close, it waits for Monday


def test_the_shares_a_waiting_sale_holds_are_not_sellable_again_and_the_cash_is_the_available_cash():
    held = holder("9000.00", reserved_cash=Decimal("1000.00"), reserved_shares={"MSFT": Decimal(8)})
    assert (held.cash_avail, held.cash_total, held.free_shares("MSFT"), held.sellable_shares("MSFT", date(2026, 11, 20)), held.position("MSFT").shares) == (Decimal("9000.00"), Decimal("10000.00"), Decimal(12), Decimal(12), Decimal(20))


# ---------------------------------------------------------------- W8: the setting

def test_the_watchers_pause_is_a_setting_of_the_account_managers_file():
    assert load_config().orders.watch_seconds == 60


# ---------------------------------------------------------------- W3: the one door to what is available (a scan of the source)

ACCOUNT_SIDE = ("prometheus/agents/account_manager", "prometheus/agents/trader", "prometheus/application", "prometheus/langgraph", "prometheus/reasoning", "prometheus/proving", "prometheus/common", "prometheus/tools")
RAW_READS = re.compile(r"cash_balance|trading\.positions|trading\.cash_ledger|FROM positions")


def test_no_module_of_the_account_side_reads_the_accounts_cash_or_holdings_except_through_the_book():
    """Every reader of what may be spent or sold reads `AccountStore.list_portfolio` / `available_cash` / `available_shares` (which take what waiting orders hold off): no module outside `persistence/accounts.py` asks the database for the cash balance,
    the positions or the cash ledger itself."""
    found = []
    for folder in ACCOUNT_SIDE:
        for path in sorted((ROOT / folder).rglob("*.py")):
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if RAW_READS.search(line) and not line.lstrip().startswith("#"):
                    found.append(f"{path.relative_to(ROOT)}:{number}")
    assert found == [], f"these read the account's cash or holdings without the door: {found}"


def test_every_function_of_the_book_that_reads_the_cash_or_the_holdings_is_either_the_door_or_reads_the_reservations_too():
    """In `persistence/accounts.py`: a method that reads `cash_balance` or `trading.positions` either adds up the reservations (`_reserved`) or is on this short list, each with its reason."""
    tree = ast.parse(inspect.getsource(accounts_module))
    source = inspect.getsource(accounts_module)
    excused = {
        "holds_fractions": "reads a column TYPE, not a quantity",
        "add_funds": "puts cash IN: a deposit can use nothing that is set aside",
        "account_balance": "the WHOLE balance, for display; what may be spent is `available_cash`",
        "history": "reads the dated trades and ledger rows for the live monitor: not a figure that decides what may be spent",
        "_lock": "the account row lock every operation takes first: it reads the balance for the caller, which adds up the reservations",
        "_cash": "the balance re-read INSIDE a booking, to check the booking against what it must have done",
        "_buy_position": "writes a position the booking has already checked",
        "_sell_position": "writes a position the booking has already checked",
        "save_decisions": "locks the account row only",
        "_ledger": "writes the cash ledger row of a booking or deposit the caller has already checked",
    }
    unread = []
    methods = [n for c in tree.body if isinstance(c, ast.ClassDef) and c.name == "AccountStore" for n in c.body if isinstance(n, ast.FunctionDef)] + [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    for node in methods:      # the methods of the store and the module's functions (a nested function belongs to the method that holds it)
        text = ast.get_source_segment(source, node) or ""
        code = "\n".join(line.split("#")[0] for line in text.splitlines())      # not the comments
        if RAW_READS.search(code) and "_reserved" not in text and node.name not in excused:
            unread.append(node.name)
    assert unread == [], f"these read the cash or the holdings and do not read the reservations: {unread}"
    for door in ("list_portfolio", "available_cash", "available_shares", "remove_funds", "place_open_order", "book_trade"):
        assert "_reserved" in inspect.getsource(getattr(accounts_module.AccountStore, door)), f"{door} must add up what waiting orders hold"


def test_only_the_account_manager_reads_the_whole_balance_and_nothing_decides_a_spend_from_it():
    """`account_balance` is the display figure (the whole cash): outside the book it is read only by the Account Manager's own `account_balance` and `format_account`, never to decide what may be spent."""
    callers = []
    for path in sorted((ROOT / "prometheus").rglob("*.py")):
        if path.name == "accounts.py" and path.parent.name == "persistence":
            continue
        text = path.read_text(encoding="utf-8")
        callers += [f"{path.relative_to(ROOT)}:{n}" for n, line in enumerate(text.splitlines(), 1) if ".account_balance(" in line and not line.lstrip().startswith("#")]
    assert {c.split(":")[0] for c in callers} <= {"prometheus/agents/account_manager/account_manager.py"}, callers


@pytest.mark.parametrize("edit, message", [
    (lambda d: d.pop("orders"), "orders"),
    (lambda d: d["orders"].pop("watch_seconds"), "watch_seconds"),
    (lambda d: d["orders"].update(watch_seconds=1), "watch_seconds"),            # under 5 seconds: not a convention a quote API can keep
    (lambda d: d["orders"].update(watch_seconds=7200), "watch_seconds"),
    (lambda d: d["orders"].update(poll=5), "poll"),                              # a typo is an error, not a default
])
def test_the_orders_section_of_the_settings_file_is_checked_like_the_others(tmp_path, edit, message):
    import json
    from prometheus.agents.account_manager import config as amc
    data = json.loads(amc.CONFIG_PATH.read_text(encoding="utf-8"))
    edit(data)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        amc.load_config(path)
