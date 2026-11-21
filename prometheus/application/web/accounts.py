"""The account routes' service (UI v1 spec 4.1 and 4.5): the user's own account in the Postgres book, created from the Profile form and remembered by number.

The page talks to the ACCOUNT MANAGER, not to the store (the user talks to the Account Manager only): `AccountManager.connect`, `users.set_name`, `set_age`, `set_trade_mode`, `set_risk_level`, `add_funds`, `remove_funds`
and `list_portfolio`. Building an AccountManager makes a model client and calls nothing. An account with a `seed_key` (the three samples) can be looked at and never written. Every refusal is a message for the user that names the thing
and never a value from `.env` or a connection detail (a database failure is its kind only, as `desk.saved_account` says it).
"""
from datetime import datetime, timezone
from decimal import Decimal

from prometheus.application.web import views
from prometheus.common.user_account import clean_age, clean_display_name, clean_risk_level, clean_trade_mode, parse_money


class ServiceError(Exception):
    """A refusal with the HTTP status it is answered with: 404 no such account, 409 a sample account is read-only, 503 the book or a key is not there."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


class Accounts:
    def __init__(self, store_factory=None, manager_factory=None):
        self._store_factory, self._manager_factory, self._store = store_factory, manager_factory, None

    # ---- the book and the Account Manager
    def store(self):
        if self._store is None:
            try:
                if self._store_factory is None:
                    from prometheus.persistence.accounts import open_account_store
                    self._store = open_account_store()
                else:
                    self._store = self._store_factory()
            except FileNotFoundError:
                raise ServiceError(503, "the account book is not set up: run scripts/setup_postgres.sh") from None
            except (ValueError, OSError) as error:
                raise ServiceError(503, f"the account book could not be opened ({type(error).__name__})") from None
        return self._store

    def book_status(self) -> str:
        """'ok', or the kind of failure (never a connection detail): for the health route."""
        try:
            self.store().ping()
            return "ok"
        except ServiceError as error:
            return error.message
        except Exception as error:
            return f"the account book could not be reached ({type(error).__name__})"

    def manager(self, account_id: int):
        try:
            if self._manager_factory is not None:
                manager = self._manager_factory()
            else:
                from prometheus.agents.account_manager.account_manager import AccountManager
                manager = AccountManager()
        except (RuntimeError, ValueError) as error:       # a missing key: the message names the setting, never a value
            raise ServiceError(503, str(error)) from None
        manager.connect(self.store(), account_id)
        return manager

    def _guard(self, fn):
        try:
            return fn()
        except ServiceError:
            raise
        except ValueError as error:
            message = str(error)
            if message.startswith("no account"):
                raise ServiceError(404, message) from None
            raise
        except Exception as error:
            raise ServiceError(503, f"the account book could not be used ({type(error).__name__})") from None

    # ---- the routes
    def view(self, account_id: int) -> dict:
        def read():
            manager = self.manager(account_id)
            return views.account_view(manager.users.get(), manager.list_portfolio(), as_of=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        return self._guard(read)

    def _writable(self, account_id: int):
        manager = self.manager(account_id)
        if manager.users.get().seed_key is not None:
            raise ServiceError(409, "sample accounts are read-only here")
        return manager

    def create(self, data: dict) -> dict:
        """A new account with its starting funds: every field is checked by the book's own `clean_*` before anything is written."""
        name, age, risk, mode = clean_display_name(data.get("name")), clean_age(data.get("age")), clean_risk_level(data.get("risk")), clean_trade_mode(data.get("mode"))
        cash = _money(data.get("cash"), "cash")

        def make():
            store = self.store()
            account_id = store.create_account(name, risk, trade_mode=mode, age=age)
            try:
                self.manager(account_id).add_funds(cash)
            except Exception:
                store.delete_account(account_id)
                raise
            return self.view(account_id)
        return self._guard(make)

    def patch(self, account_id: int, data: dict) -> dict:
        known = {"name", "age", "risk", "mode"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"cannot change {', '.join(sorted(unknown))}")

        def change():
            manager = self._writable(account_id)
            if "name" in data:
                manager.users.set_name(data["name"])
            if "age" in data:
                manager.users.set_age(data["age"])
            if "mode" in data:
                manager.users.set_trade_mode(data["mode"])
            if "risk" in data:
                manager.set_risk_level(data["risk"])
            return self.view(account_id)
        return self._guard(change)

    def funds(self, account_id: int, action: str, amount) -> dict:
        if action not in ("add", "remove"):
            raise ValueError("action must be add or remove")
        money = _money(amount, "amount")

        def move():
            manager = self._writable(account_id)
            (manager.add_funds if action == "add" else manager.remove_funds)(money)
            return self.view(account_id)
        return self._guard(move)

    def orders(self, account_id: int) -> dict:
        """The account's WAITING limit orders (design choices entry 140): `{orders: [...], reserved_cash, available, watching}`. `watching` says whether a watcher (a server) holds the advisory lock now: an order is watched only while the page's server runs."""
        def read():
            manager = self.manager(account_id)
            held = manager.list_portfolio()
            return {"orders": [o.as_dict() for o in manager.open_orders()], "reserved_cash": held.reserved_cash, "available": held.cash_avail, "watching": bool(self.store().watcher_running())}
        return self._guard(read)

    def cancel_order(self, order_id: int, when=None) -> dict:
        """The user cancels a waiting order by its number: the account it belongs to is found by the book's own function (`account_of_open_order`), the account must be writable (a sample is read-only), and what the order set aside is available again.
        Returns the cancelled order. 404 for an order that does not exist, 409 for one that is not open (filled, expired, cancelled or rejected)."""
        def cancel():
            account_id = self.store().account_of_open_order(order_id)
            if account_id is None:
                raise ServiceError(404, f"there is no order {order_id}")
            manager = self._writable(account_id)
            try:
                return manager.cancel_open_order(order_id, when).as_dict()
            except ValueError as error:
                raise ServiceError(409, str(error)) from None
        return self._guard(cancel)

    def writable(self, account_id: int):
        """The holder of an account a run may use. Raises ServiceError for an account that does not exist or is a sample (a run on one is refused the same way as a write)."""
        return self._guard(lambda: self._writable(account_id).users.get())


def _money(value, label: str) -> Decimal:
    if not isinstance(value, str):
        raise ValueError(f'{label} must be a string of dollars such as "500.00" (a number is refused)')
    return parse_money(value)
