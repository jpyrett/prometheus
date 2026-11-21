"""Shared setup of the tests of the WAITING LIMIT ORDERS (design choices entry 140; design/limit_orders_spec.md, Part B). REAL data, no stand-in agent and no stand-in model:

    store, db                       the real local Postgres on THROWAWAY accounts the test creates and deletes (the three sample accounts and the kept paper accounts are never touched: a check at the end of every test)
    stored_close                    the real Data Clerk answers a quote with the STORED market close of a day (USE_LIVE_TRADING=false: the stored daily bars, `data/historical/day_aggs_v1`, read only; Finnhub is never asked): the test
                                    chooses the day, so a price that "falls" is the real close of a later session of the same stock
    eastern(day, hour, minute)      a moment in the market's own time zone (the fake clock of the watcher: the real calendar's sessions)

The real closes used (read from the stored bars, not typed here): XOM 2026-11-13 170.59, 2026-11-16 166.87, 2026-11-17 167.21, 2026-11-18 160.58; MSFT 2026-11-12 523.00, 2026-11-16 517.99; MU 2026-11-16 719.02, 2026-11-18 705.27.
A machine without those stored bars skips the tests that need them.
"""
import json
from contextlib import suppress
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from prometheus.agents.account_manager.account_manager import AccountManager
from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
from prometheus.agents.trader.trader import NullLedger, Trader
from prometheus.common.money import price as price_door
from prometheus.persistence.accounts import SECRETS, open_account_store

SAMPLES = ("jacob_sample", "avery_chen", "blake_okonkwo")
YEAR = (date(2020, 1, 1), date(2030, 12, 31))
MONDAY = date(2026, 11, 16)
FRIDAY = date(2026, 11, 13)
NEW_YORK = ZoneInfo("America/New_York")


def eastern(day: date, hour: int, minute: int = 0) -> datetime:
    """A moment on `day` at this clock time in New York (an aware datetime)."""
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=NEW_YORK)


def clerk() -> DataClerk:
    return DataClerk(model=object())      # its quotes and bars are code: its model is never called


def close_of(symbol: str, day: date) -> Decimal:
    """The real stored close of `symbol` on `day`, as the Account Manager takes a quote (through the price door)."""
    return price_door(clerk().tools.close_quote(symbol, day)["price"])


def stored_closes_available() -> bool:
    try:
        return all(close_of(symbol, day) for symbol, day in (("XOM", MONDAY), ("XOM", date(2026, 11, 18)), ("MSFT", date(2026, 11, 12)), ("MSFT", MONDAY), ("MU", MONDAY)))
    except Exception:
        return False


@pytest.fixture
def stored_close(monkeypatch):
    """The day whose stored close the real Data Clerk answers a quote with: `stored_close["day"]` (Monday 2026-11-16 to start with; a test sets another day to make a price move). Skips when the stored bars are not on this machine."""
    monkeypatch.setenv("USE_LIVE_TRADING", "false")
    state = {"day": MONDAY}
    monkeypatch.setattr("prometheus.agents.data_clerk.data_clerk_agent.market_day", lambda *a, **k: state["day"])
    if not stored_closes_available():
        pytest.skip("the stored daily bars of XOM, MSFT and MU for November 2026 are not on this machine (data/historical/day_aggs_v1)")
    return state


@pytest.fixture(scope="module")
def store():
    if not SECRETS.is_file():
        pytest.skip("secrets/postgres_local.txt is not on this machine (run scripts/setup_postgres.sh)")
    try:
        opened = open_account_store()
        opened.ping()
        with opened._connection() as conn:
            if conn.execute("SELECT to_regclass('trading.open_orders') IS NOT NULL AS there").fetchone()["there"] is not True:
                pytest.skip("the account book has no table for waiting orders: run `bash scripts/setup_postgres.sh`")
    except pytest.skip.Exception:
        raise
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
            state[seed] = (self.store.account_balance(account_id), self.store.list_portfolio(account_id).model_dump_json(), self.store.number_trades(account_id, *YEAR), self.store.open_orders(account_id, ("open", "filled", "expired", "cancelled", "rejected")))
        return state

    def account(self, cash: str, name: str = "Waiting Orders Holder", mode: str = "ask") -> int:
        """A new throwaway account with this much cash (a string of dollars): deleted when the test ends."""
        account_id = self.store.create_account(name, "moderate", trade_mode=mode)
        self.made.append(account_id)
        self.store.add_funds(account_id, Decimal(cash), date(2026, 11, 13))
        return account_id

    def buy(self, account_id: int, symbol: str, shares: int, price: str, day: date = date(2026, 10, 1)) -> None:
        """Hold `shares` of `symbol` from long ago (booked through the book's own transaction): a position a sale can sell."""
        self.store.book_trade(account_id, symbol, "buy", Decimal(shares), Decimal(price), day)

    def manager(self, account_id: int, **options) -> AccountManager:
        """The real Account Manager over the real Data Clerk and this account (no model call is made)."""
        options.setdefault("clerk", clerk())
        return AccountManager(model=object(), model_id="open-orders-test", store=self.store, account_id=account_id, **options)


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


FIXTURES = Path(__file__).parent / "fixtures"
DIRECTORY = json.loads((FIXTURES / "finnhub_company_search.json").read_text(encoding="utf-8"))["queries"]      # Finnhub's REAL recorded answers of the symbol directory (made 2026-11-15)


def directory(query: str):
    """The symbol directory: Finnhub's REAL recorded answers. A query nobody recorded is a failure of the test, not an empty answer."""
    if query not in DIRECTORY:
        raise AssertionError(f"the Data Clerk asked the symbol directory for {query!r}, which was not recorded")
    return DIRECTORY[query]


def clerk_with_directory() -> DataClerk:
    """The real Data Clerk over the recorded directory (its prices are the stored closes of `stored_close`): what a direct order needs to find the company and the symbol of what the user named."""
    real = DataClerk(model=object(), symbol_search=directory)
    real.lookup_interval = 0
    return real


def trader(**options) -> Trader:
    """The real Trader with no model (the sweep places the order) and a ledger that keeps nothing."""
    return Trader(use_model=False, ledger=NullLedger(), **options)
