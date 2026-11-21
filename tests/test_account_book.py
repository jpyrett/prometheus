"""The account book: funds, holdings, risk, realized gain, and the recommendation counts (design choices entry 72).

The checks of money, dates and risk levels do not need a database. The rest uses the local Postgres from scripts/setup_postgres.sh.
No Finnhub call and no model. If the password file is missing the database tests are skipped; if it is present and Postgres is down, they fail.
"""
from contextlib import contextmanager
from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from prometheus.agents.account_manager import account_manager as am
from prometheus.common import work_order as wo
from prometheus.common.user_account import clean_risk_level, date_window, parse_money
from prometheus.persistence.accounts import SECRETS, open_account_store

YEAR = (date(2026, 1, 1), date(2026, 12, 31))


def test_money_dates_and_risk_levels_are_checked_before_any_database():
    assert parse_money(Decimal("100")) == Decimal("100.00")
    assert parse_money("1,000.50") == Decimal("1000.50")
    assert parse_money(15) == Decimal("15.00")
    for bad in (0, -1, Decimal("10.001"), "1e2", 1.5, True, "ten", None):
        with pytest.raises(ValueError):
            parse_money(bad)
    with pytest.raises(ValueError):
        date_window(date(2026, 5, 2), date(2026, 5, 1))
    with pytest.raises(ValueError):
        date_window(datetime(2026, 5, 1, 12, 0), date(2026, 5, 2))
    assert clean_risk_level(" AGGRESSIVE ") == "aggressive"
    for level in ("low", "medium", "high", "very_high", "yolo"):
        with pytest.raises(ValueError, match="conservative, moderate, aggressive"):
            clean_risk_level(level)


def test_without_a_database_the_profile_is_still_the_json_file():
    manager = am.AccountManager(model=object(), model_id="account-book")
    assert (manager.store, manager.account_id, manager.profile.user, manager.profile.risk_tolerance) == (None, None, "Jacob Pyrett", "moderate")
    with pytest.raises(ValueError, match="saved account"):
        manager.set_risk_level("aggressive")
    assert manager.profile.risk_tolerance == "moderate"
    with pytest.raises(ValueError, match="together"):
        am.AccountManager(model=object(), model_id="account-book", store=object())


def test_the_command_needs_a_purchase_or_an_account(capsys):
    assert am.main([]) == 1
    assert capsys.readouterr().err.startswith("ERROR")


@pytest.fixture(scope="module")
def store():
    if not SECRETS.is_file():
        pytest.skip("secrets/postgres_local.txt is not on this machine (run scripts/setup_postgres.sh)")
    try:
        opened = open_account_store()
        opened.ping()
    except Exception as error:
        pytest.fail(f"Postgres is configured but not reachable: {error}")
    return opened


@contextmanager
def book(store, name="Test Holder", risk="moderate"):
    account_id = store.create_account(name, risk)
    try:
        yield am.AccountManager(model=object(), model_id="account-book", store=store, account_id=account_id)
    finally:
        store.delete_account(account_id)


def test_the_sample_accounts_match_the_seed(store):
    jacob = am.AccountManager(model=object(), model_id="account-book", store=store, account_id=store.account_id_for_seed("jacob_sample"))
    avery = am.AccountManager(model=object(), model_id="account-book", store=store, account_id=store.account_id_for_seed("avery_chen"))
    blake = am.AccountManager(model=object(), model_id="account-book", store=store, account_id=store.account_id_for_seed("blake_okonkwo"))

    assert jacob.account_balance() == Decimal("51.23")
    assert jacob.profile.risk_tolerance == "moderate" and jacob.profile.trade_mode == "ask" and jacob.profile.user == "Jacob Pyrett"
    assert {p.symbol: p.shares for p in jacob.list_portfolio().positions} == {"GM": 60, "MSFT": 50, "TSLA": 123}
    assert (jacob.calculate_total_gain(*YEAR), jacob.number_trades(*YEAR)) == (Decimal("0.00"), 0)
    assert (jacob.number_approved_recommendations(*YEAR), jacob.number_denied_recommendations(*YEAR)) == (1, 1)

    assert avery.account_balance() == Decimal("4785.00")
    assert avery.users.get().email == "avery.chen@example.com" and avery.profile.risk_tolerance == "conservative"
    assert [(p.symbol, p.shares) for p in avery.list_portfolio().positions] == [("F", 15)]
    assert avery.calculate_total_gain(*YEAR) == Decimal("-27.50")
    assert avery.calculate_total_gain(date(2026, 5, 1), date(2026, 5, 31)) == Decimal("12.50")
    assert avery.calculate_total_gain(date(2026, 7, 1), date(2026, 7, 1)) == Decimal("-40.00")
    assert avery.number_trades(*YEAR) == 4 and avery.number_trades(date(2026, 3, 1), date(2026, 3, 31)) == 1
    assert (avery.number_approved_recommendations(*YEAR), avery.number_denied_recommendations(*YEAR)) == (2, 1)
    assert avery.number_approved_recommendations(date(2026, 8, 1), date(2026, 8, 31)) == 0
    assert avery.number_denied_recommendations(date(2026, 8, 1), date(2026, 8, 31)) == 0

    assert blake.account_balance() == Decimal("1800.00") and blake.profile.risk_tolerance == "aggressive" and blake.profile.trade_mode == "auto"
    assert [(p.symbol, p.shares) for p in blake.list_portfolio().positions] == [("PFE", 8)]
    assert (blake.calculate_total_gain(*YEAR), blake.number_trades(*YEAR)) == (Decimal("0.00"), 1)
    assert (blake.number_approved_recommendations(*YEAR), blake.number_denied_recommendations(*YEAR)) == (0, 2)
    with pytest.raises(ValueError, match="cannot be deleted"):
        store.delete_account(jacob.account_id)
    assert jacob.account_balance() == Decimal("51.23")


def test_a_new_account_takes_funds_lists_holds_and_counts_the_gain(store):
    with book(store) as manager:
        assert manager.account_balance() == Decimal("0.00")
        assert manager.calculate_total_gain(*YEAR) == Decimal("0.00") and manager.list_portfolio().positions == []
        assert manager.add_funds(Decimal("1000.50"), date(2026, 1, 15)) == Decimal("1000.50")
        assert manager.add_funds("10") == Decimal("1010.50")
        with pytest.raises(ValueError):
            manager.add_funds(1.25)
        assert manager.account_balance() == Decimal("1010.50")
        assert manager.calculate_total_gain(*YEAR) == Decimal("0.00")
        assert manager.set_risk_level("aggressive") == "aggressive"
        assert manager.profile.risk_tolerance == "aggressive" and manager.risk_view().risk_tolerance == "aggressive"
        with pytest.raises(ValueError, match="conservative, moderate, aggressive"):
            manager.set_risk_level("low")
        assert manager.profile.risk_tolerance == "aggressive"
        holder = manager.users.set_name("Test Holder Renamed")
        holder = manager.users.set_email("holder@example.com")
        holder = manager.users.set_preferred_name("Hold")
        holder = manager.users.set_phone("+1 412 555 0100")
        holder = manager.users.set_timezone("America/Chicago")
        holder = manager.users.set_note("gui test")
        assert holder.account_number == manager.account_id
        assert (holder.display_name, holder.email, holder.preferred_name, holder.phone, holder.timezone, holder.note) == (
            "Test Holder Renamed", "holder@example.com", "Hold", "+1 412 555 0100", "America/Chicago", "gui test")
        assert manager.list_portfolio().user == "Test Holder Renamed"
        with pytest.raises(ValueError):
            manager.users.set_email("not-an-email")
        assert manager.users.get().email == "holder@example.com"
        assert manager.record_trade("F", "buy", 10, Decimal("12.50"), date(2026, 2, 2), name="Ford Motor Co") == Decimal("0.00")
        assert manager.account_balance() == Decimal("885.50")
        portfolio = manager.list_portfolio()
        assert [(p.symbol, p.shares, p.cost_basis) for p in portfolio.positions] == [("F", 10, Decimal("12.5"))]
        assert manager.record_trade("F", "sell", 4, Decimal("15"), date(2026, 5, 20)) == Decimal("10.00")
        assert manager.account_balance() == Decimal("945.50")
        assert manager.list_portfolio().position("F").shares == 6
        assert manager.calculate_total_gain(*YEAR) == Decimal("10.00")
        assert manager.calculate_total_gain(date(2026, 2, 1), date(2026, 2, 28)) == Decimal("0.00")
        assert manager.number_trades(*YEAR) == 2
        with pytest.raises(ValueError, match="does not cover"):
            manager.record_trade("GM", "buy", 1000, Decimal("40"), date(2026, 6, 1))
        with pytest.raises(ValueError, match="holds 6"):
            manager.record_trade("F", "sell", 7, Decimal("15"), date(2026, 6, 1))
        assert manager.account_balance() == Decimal("945.50") and manager.number_trades(*YEAR) == 2
        manager.users.set_status("suspended")
        with pytest.raises(ValueError, match="suspended"):
            manager.add_funds(Decimal("1.00"))
        assert manager.account_balance() == Decimal("945.50")


def test_fills_the_trader_reports_are_booked_through_notify_trades_and_the_book_and_the_portfolio_agree(store):
    """Trader v1 spec section 13 on a THROWAWAY account in the real local Postgres: deposit, book a buy and a sell through `notify_trades(..., book=True)`, then check the cash, the position and the realized gain.
    The three sample accounts are read before and after and must not move."""
    from test_account_manager_trades import filled

    samples = ("jacob_sample", "avery_chen", "blake_okonkwo")

    def sample_state():
        state = {}
        for seed in samples:
            account_id = store.account_id_for_seed(seed)
            state[seed] = (store.account_balance(account_id), store.list_portfolio(account_id).model_dump_json(), store.number_trades(account_id, *YEAR))
        return state

    before = sample_state()
    bought_on, sold_on = date(2026, 11, 5), date(2026, 11, 6)
    with book(store, name="Trader Holder") as manager:
        assert manager.add_funds(Decimal("2000.00"), bought_on) == Decimal("2000.00")
        portfolio = manager.trading_portfolio()                                                    # connected: the saved account, not a file
        assert portfolio.user == "Trader Holder" and portfolio.cash_avail == Decimal("2000") and portfolio.positions == []

        after_buy, problems = manager.notify_trades(portfolio, [filled("T1", "PG", "buy", 10, Decimal("150.25"), record_id="rec-buy")], bought_on, book=True)
        assert problems == [] and manager.account_balance() == Decimal("497.50")                   # 2000.00 - round(10 x 150.25, 2)
        saved = manager.list_portfolio()
        assert [(p.symbol, p.shares, p.cost_basis, p.acquired) for p in saved.positions] == [("PG", 10, Decimal("150.25"), bought_on)]
        assert after_buy.cash_avail == Decimal("497.5") and after_buy.position("PG").shares == 10 and after_buy.position("PG").cost_basis == Decimal("150.25")
        assert manager.calculate_total_gain(bought_on, sold_on) == Decimal("0.00") and manager.number_trades(bought_on, sold_on) == 1

        after_sell, problems = manager.notify_trades(saved, [filled("T2", "PG", "sell", 4, Decimal("160.0"), record_id="rec-sell")], sold_on, book=True)
        assert problems == [] and manager.account_balance() == Decimal("1137.50")                  # 497.50 + 640.00
        saved = manager.list_portfolio()
        assert [(p.symbol, p.shares, p.cost_basis) for p in saved.positions] == [("PG", 6, Decimal("150.25"))]
        assert manager.calculate_total_gain(bought_on, sold_on) == Decimal("39.00")                # 640.00 - round(4 x 150.25, 2) = 601.00
        assert manager.calculate_total_gain(bought_on, bought_on) == Decimal("0.00") and manager.number_trades(bought_on, sold_on) == 2
        assert after_sell.cash_avail == manager.account_balance() and after_sell.position("PG").shares == saved.position("PG").shares     # the portfolio returned is what the book holds

        stale = saved.model_copy(update={"cash_avail": Decimal("10000")})                                    # a portfolio the book does not agree with: the book refuses what the cash cannot pay for
        unchanged, problems = manager.notify_trades(stale, [filled("T3", "F", "buy", 20, Decimal("100.0"), record_id="rec-big")], sold_on, book=True)
        assert problems == ["T3: buy 20 F was not booked: cash $1,137.50 does not cover $2,000.00"] and unchanged == stale
        assert manager.account_balance() == Decimal("1137.50") and manager.number_trades(bought_on, sold_on) == 2 and manager.list_portfolio().position("F") is None

        not_booked, problems = manager.notify_trades(saved, [filled("T4", "PG", "sell", 1, Decimal("170.0"), record_id="rec-memory")], sold_on)       # no book=True: nothing is saved
        assert problems == [] and not_booked.position("PG").shares == 5 and manager.account_balance() == Decimal("1137.50") and manager.list_portfolio().position("PG").shares == 6
    assert sample_state() == before


def test_the_account_command_prints_the_samples(store, capsys):
    assert am.main(["--account", "avery_chen", "--from", "2026-01-01", "--to", "2026-12-31"]) == 0
    text = capsys.readouterr().out
    assert "Avery Chen" in text and "realized gain -27.50" in text and "trades 4" in text and "approved 2" in text and "denied 1" in text and "F" in text
    assert am.main(["--account", "jacob_sample", "--from", "2024-01-01", "--to", "2026-12-31"]) == 0
    text = capsys.readouterr().out
    assert "Jacob Pyrett" in text and "cash 51.23" in text and "MSFT" in text and "moderate" in text
    assert am.main(["--account", "blake_okonkwo", "--from", "2026-01-01", "--to", "2026-12-31"]) == 0
    text = capsys.readouterr().out
    assert "Blake Okonkwo" in text and "cash 1,800.00" in text and "aggressive" in text and "PFE" in text and "denied 2" in text
    assert am.main(["--account", "no_such_sample"]) == 1
    assert "ERROR" in capsys.readouterr().err
