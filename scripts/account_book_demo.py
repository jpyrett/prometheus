"""Print the three sample accounts, then book a throwaway account and remove it.

No Finnhub call and no model. The database must already be up (scripts/setup_postgres.sh).

    python scripts/account_book_demo.py
"""
import sys
from datetime import date
from decimal import Decimal

from prometheus.agents.account_manager.account_manager import AccountManager, format_account
from prometheus.persistence.accounts import open_account_store

WINDOW = (date(2026, 1, 1), date(2026, 12, 31))


def expect(label, got, want):
    if got != want:
        raise SystemExit(f"{label}: got {got}, want {want}")


def show(manager):
    print(format_account(manager, *WINDOW))
    print()


def main() -> int:
    store = open_account_store()
    print(store.ping())
    print()
    samples = {}
    for key in ("jacob_sample", "avery_chen", "blake_okonkwo"):
        manager = AccountManager(model=object(), model_id="account-book", store=store, account_id=store.account_id_for_seed(key))
        samples[key] = manager.account_balance()
        show(manager)

    account_id = store.create_account("Demo Trader", "moderate", preferred_name="Demo", email="demo.trader@example.com")
    try:
        manager = AccountManager(model=object(), model_id="account-book", store=store, account_id=account_id)
        expect("opening cash", manager.account_balance(), Decimal("0.00"))
        expect("after deposit", manager.add_funds(Decimal("250.00"), date(2026, 2, 1)), Decimal("250.00"))
        expect("risk", manager.set_risk_level("conservative"), "conservative")
        manager.users.set_phone("+1 412 555 0100")
        expect("buy gain", manager.record_trade("F", "buy", 4, Decimal("10.00"), date(2026, 2, 2), name="Ford Motor Co"), Decimal("0.00"))
        expect("sell gain", manager.record_trade("F", "sell", 4, Decimal("12.50"), date(2026, 3, 2)), Decimal("10.00"))
        expect("cash", manager.account_balance(), Decimal("260.00"))
        expect("gain", manager.calculate_total_gain(*WINDOW), Decimal("10.00"))
        expect("trades", manager.number_trades(*WINDOW), 2)
        held = manager.list_portfolio()
        expect("flat", held.positions, [])
        expect("profile", manager.profile.risk_tolerance, "conservative")
        print(format_account(manager, *WINDOW))
        print()
    finally:
        store.delete_account(account_id)
    for key, balance in samples.items():
        manager = AccountManager(model=object(), model_id="account-book", store=store, account_id=store.account_id_for_seed(key))
        expect(key, manager.account_balance(), balance)
    print("demo account removed; the three sample balances are unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
