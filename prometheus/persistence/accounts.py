"""The account book in Postgres (design choices entry 72). Plain Python: no model.

    store = open_account_store()                          # reads secrets/postgres_local.txt
    account_id = store.account_id_for_seed("avery_chen")
    store.add_funds(account_id, Decimal("100.00"))
    store.calculate_total_gain(account_id, date(2026, 1, 1), date(2026, 12, 31))

Realized gain is the profit booked on sells whose trade date is inside the window, including both ends:
round(price times shares, 2) minus round(the average cost times shares, 2). A buy is 0. A deposit is not a gain.
Shares still held are not marked to a market price, because that would need a price on each date.

The application role can read a row only after this process sets `trading.account_id` for the transaction.
Sample accounts (a seed_key) are not deleted through this class.

SHARES MAY HAVE A FRACTION (design choices entry 110). `trading.positions.shares`, `positions.recent_shares`, `trades.shares` and `recommendation_decisions.shares` are `numeric(18,6)` once `scripts/setup_postgres.sh` has applied
sql/prometheus_accounts.sql (they were `integer`); `holds_fractions()` says which this database has. A share quantity comes into `book_trade` and `record_trade` as `shares.quantity` (a Decimal with at most six
decimals) and is held as a `Decimal` at scale 6 (`shares.as_decimal`) from the door on; a quantity with a fraction on a book that does not hold fractions is refused before anything is written, with the fix named, and is
never rounded into an integer column. What the book returns (`Booked`, `list_portfolio`) is a quantity again, in its canonical form (a whole number has no decimals).

WAITING ORDERS AND WHAT IS SET ASIDE (design choices entry 140; Jacob, 2026-11-21). A direct order whose limit price is not met waits as a DAY order (`trading.open_orders`) and SETS ASIDE what it needs: a purchase its most possible
cost in cash, a sale its shares. Set-aside cash and shares are NOT AVAILABLE to anything else, and this module is the ONE DOOR to what is available: `available_cash(account_id)` is the cash less the cash the account's open orders hold,
`available_shares(account_id, symbol)` the shares held less the shares they hold, and `list_portfolio` (what every other reader of the account's cash and holdings reads) returns the AVAILABLE cash as `cash_avail` and the whole holding with
the reservations beside it (`Portfolio.reserved_cash`, `reserved_shares`). `remove_funds`, `place_open_order` and `book_trade` read the same reservations under the same row locks, so a purchase, a sale, a withdrawal or another waiting
order that would use what is set aside is refused by the book itself, whatever a caller read before. `trading.account_id` row security applies to the new table like the others; `accounts_with_open_orders` and `account_of_open_order` (SECURITY
DEFINER, they return an id and nothing else) are the only two questions asked across accounts.

MONEY, PRICES AND SHARES ARE `decimal.Decimal` ALL THE WAY (design choices entry 112; design/decimal_money_spec.md): what the book reads from Postgres is a Decimal and stays one (`list_portfolio`, `account_balance`, `Booked`, the
gain), and what it is given is a Decimal already: a price goes through `parse_price` (a float or a bool is refused, it comes in through a door once), an amount through `parse_money`, a quantity through `shares.quantity`. Nothing here turns a Decimal into an
int or a float, and nothing here turns a float into a Decimal: the one exception is the profile's own years and minimum price (`load_profile`, `_user`), which are the Advisor's and the Critic's measurements (D6), not the account's money.
"""
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import psycopg
from psycopg import errors
from psycopg.rows import dict_row

from prometheus.common import shares as sh
from prometheus.common.money import EXACT
from prometheus.common.portfolio import Portfolio, trade_amount
from prometheus.common.user_account import (
    UserAccount,
    cents,
    clean_age,
    clean_display_name,
    clean_email,
    clean_horizon,
    clean_note,
    clean_optional_name,
    clean_phone,
    clean_risk_level,
    clean_status,
    clean_timezone,
    clean_trade_mode,
    date_window,
    parse_money,
    parse_price,
    price_scale,
)
from prometheus.common.user_profile import UserProfile
from prometheus.rag.loaders import check_symbol

REPO_ROOT = Path(__file__).resolve().parents[2]
SECRETS = REPO_ROOT / "secrets" / "postgres_local.txt"
_KEYS = ("host", "port", "database", "app_user", "app_password")
FRACTION_REFUSED = ("this account book holds whole shares only, so a fraction of a share was not written: its share columns are still integer. Apply the new schema with `bash scripts/setup_postgres.sh` "
                    "(it keeps every table and account), or trade whole shares")
BOOKING_LOCK_SECONDS = 10      # how long a booking waits for the locks on the account's portfolio before it fails (nothing is committed); a stuck lock must not hang a run
APPROVED_DECISIONS = ("yes_auto", "yes_user")
DENIED_DECISIONS = ("no_system_denied", "no_user_denied", "no_expired", "no_funds")
PENDING_DECISIONS = ("pending_user",)
DECISION_STATES = APPROVED_DECISIONS + DENIED_DECISIONS + PENDING_DECISIONS
OPEN_ORDER_STATUSES = ("open", "filled", "expired", "cancelled", "rejected")      # `open` holds what it set aside; the other four are final and hold nothing
WATCHER_LOCK_KEY = 1_150_920_113      # the Postgres advisory lock the web server's watcher holds: one watcher for each database, a second server does not double-check the orders
_EDITABLE = {
    "display_name": clean_display_name,
    "preferred_name": clean_optional_name,
    "email": clean_email,
    "phone": clean_phone,
    "timezone": clean_timezone,
    "note": clean_note,
    "status": clean_status,
    "trade_mode": clean_trade_mode,
    "age": clean_age,
    "investment_horizon_years": clean_horizon,
    "risk_tolerance": clean_risk_level,
}


def load_db_config(path: Path | None = None) -> dict[str, str]:
    """The local connection file. Raises FileNotFoundError or ValueError, and never prints a password."""
    path = path or SECRETS
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist (run scripts/setup_postgres.sh)")
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    missing = [key for key in _KEYS if not values.get(key)]
    if missing:
        raise ValueError(f"{path} is missing {', '.join(missing)}")
    return values


def open_account_store(path: str | Path | None = None) -> "AccountStore":
    return AccountStore(load_db_config(Path(path) if path else None))


class AccountStore:
    def __init__(self, config: dict[str, str]):
        self.config = config
        self._connect = {
            "host": config["host"], "port": int(config["port"]), "dbname": config["database"],
            "user": config["app_user"], "password": config["app_password"],
            "sslmode": config.get("sslmode") or "require", "connect_timeout": 5,
            "application_name": "prometheus_account_manager",
            "options": "-c statement_timeout=10000 -c search_path=trading",
        }
        self._fractions = False                    # `holds_fractions`: True once the share columns were seen to be numeric
        self._orders_table = False                 # True once `trading.open_orders` was seen (a book that `bash scripts/setup_postgres.sh` has not been run on since entry 140 has none, and holds nothing for anyone)

    def __repr__(self) -> str:
        return f"AccountStore(host={self.config['host']!r}, database={self.config['database']!r}, user={self.config['app_user']!r})"

    def holds_fractions(self) -> bool:
        """Whether this account book can hold a fraction of a share: the type of `trading.positions.shares` is `numeric` (it is `integer` in a database that sql/prometheus_accounts.sql of before entry 110 made). Read from
        `information_schema.columns`; a True is remembered (the type is never changed back), a False is read again next time, so that a server that was started before `bash scripts/setup_postgres.sh` sees the change."""
        if self._fractions:
            return True
        def fn(cur):
            cur.execute("""SELECT data_type FROM information_schema.columns
                           WHERE table_schema = 'trading' AND table_name = 'positions' AND column_name = 'shares'""")
            row = cur.fetchone()
            return row is not None and row["data_type"] == "numeric"
        self._fractions = bool(self._run(None, fn))
        return self._fractions

    def ping(self) -> str:
        with self._connection() as conn:
            row = conn.execute("SELECT current_user AS user_name, current_setting('server_version') AS version").fetchone()
        return f"{row['user_name']} on PostgreSQL {row['version']}"

    def account_id_for_seed(self, seed_key: str) -> int:
        def fn(cur):
            cur.execute("SELECT trading.account_id_for_seed(%s) AS account_id", (seed_key,))
            found = cur.fetchone()["account_id"]
            if found is None:
                raise ValueError(f"no sample account {seed_key!r}")
            return found
        return self._run(None, fn)

    def create_account(self, display_name: str, risk_tolerance: str, *, preferred_name: str | None = None, email: str | None = None,
                       phone: str | None = None, timezone: str = "America/New_York", trade_mode: str = "ask", note: str | None = None,
                       age: int | None = None, investment_horizon_years=None) -> int:
        values = (clean_display_name(display_name), clean_risk_level(risk_tolerance), clean_optional_name(preferred_name), clean_email(email),
                  clean_phone(phone), clean_timezone(timezone), clean_trade_mode(trade_mode), clean_note(note), clean_age(age), clean_horizon(investment_horizon_years))
        def fn(cur):
            cur.execute("SELECT trading.create_account(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) AS account_id", values)
            return cur.fetchone()["account_id"]
        return self._run(None, fn)

    def delete_account(self, account_id: int) -> None:
        def fn(cur):
            cur.execute("SELECT trading.delete_account(%s)", (account_id,))
        self._run(account_id, fn)

    def get_user(self, account_id: int) -> UserAccount:
        return _user(self._account_row(account_id))

    def load_profile(self, account_id: int) -> UserProfile:
        row = self._account_row(account_id)
        return UserProfile(
            user=row["display_name"], risk_tolerance=row["risk_tolerance"], age=row["age"],
            investment_horizon_years=None if row["investment_horizon_years"] is None else float(row["investment_horizon_years"]),
            trade_mode=row["trade_mode"],
            min_price=float(row["min_price"]), excluded_symbols=list(row["excluded_symbols"]), excluded_sectors=list(row["excluded_sectors"]),
            note=row["note"])

    def update_user(self, account_id: int, **changes) -> UserAccount:
        unknown = set(changes) - set(_EDITABLE)
        if unknown:
            raise ValueError(f"cannot change {', '.join(sorted(unknown))}")
        cleaned = {key: _EDITABLE[key](value) for key, value in changes.items()}
        if not cleaned:
            return self.get_user(account_id)
        assignments = ", ".join(f"{key} = %s" for key in cleaned)
        values = [cleaned[key] for key in cleaned]

        def fn(cur):
            self._lock(cur, account_id)
            cur.execute(f"UPDATE trading.accounts SET {assignments} WHERE account_id = %s", (*values, account_id))
            if cur.rowcount != 1:
                raise ValueError(f"no account {account_id}")
        self._run(account_id, fn)
        return self.get_user(account_id)

    def set_risk_level(self, account_id: int, level: str) -> str:
        cleaned = clean_risk_level(level)
        self.update_user(account_id, risk_tolerance=cleaned)
        return cleaned

    def add_funds(self, account_id: int, amount, when: date | None = None) -> Decimal:
        money = parse_money(amount)
        when = when or date.today()
        if type(when) is not date:
            raise ValueError("the deposit date must be a date")

        def fn(cur):
            row = self._lock(cur, account_id)
            _require_active(row)
            self._ledger(cur, account_id, "deposit", money, cents(row["cash_balance"] + money), when, "deposit")
            return self._cash(cur, account_id)
        return self._run(account_id, fn)

    def remove_funds(self, account_id: int, amount, when: date | None = None) -> Decimal:
        """Take settled cash out of the account (a withdrawal): the account row is locked, must be active, and the AVAILABLE cash (the balance less what waiting orders have set aside) must hold the amount. One `cash_ledger` row of kind `withdrawal`
        with a negative amount and the balance that follows. Returns the cash balance after it. This is the cash check the book already makes, not a limit on a trade (design choices entry 100)."""
        money = parse_money(amount)
        when = when or date.today()
        if type(when) is not date:
            raise ValueError("the withdrawal date must be a date")

        def fn(cur):
            row = self._lock(cur, account_id)
            _require_active(row)
            held, _ = self._reserved(cur, account_id, lock=True)      # the cash a waiting order has set aside cannot be taken out (design choices entry 140)
            if money > row["cash_balance"] - held:
                raise ValueError(f"the account has ${row['cash_balance']:,.2f} in cash: ${money:,.2f} cannot be removed" if not held else
                                 f"the account has ${row['cash_balance']:,.2f} in cash and ${held:,.2f} of it is set aside for waiting orders: only ${row['cash_balance'] - held:,.2f} can be removed, not ${money:,.2f}")
            self._ledger(cur, account_id, "withdrawal", -money, cents(row["cash_balance"] - money), when, "withdrawal")
            return self._cash(cur, account_id)
        return self._run(account_id, fn)

    def account_balance(self, account_id: int) -> Decimal:
        """The WHOLE cash balance of the account, what the ledger holds (what the user sees as the cash). What may be SPENT is `available_cash`: this less what waiting orders have set aside (design choices entry 140)."""
        def fn(cur):
            return self._cash(cur, account_id)
        return self._run(account_id, fn)

    def list_portfolio(self, account_id: int, release: int | None = None) -> Portfolio:
        """THE reader of the account's cash and holdings for everything that decides what may be spent or sold (design choices entry 140). `cash_avail` is the AVAILABLE cash: the cash balance less what the account's open orders
        have set aside; `positions` is the whole holding, and `reserved_cash` and `reserved_shares` say what the open orders hold (`Portfolio.sellable_shares` and `free_shares` take the shares a waiting sale holds off). `release` is the
        number of ONE open order whose own hold is left out: what the order being filled may use (the Account Manager passes it only while it fills that order). Without waiting orders it is the cash and the holdings, as ever."""
        def fn(cur):
            cur.execute("SELECT display_name, currency, cash_balance, note FROM trading.accounts WHERE account_id = %s", (account_id,))
            account = cur.fetchone()
            if account is None:
                raise ValueError(f"no account {account_id}")
            cur.execute("""SELECT symbol, name, shares, cost_basis, acquired, recent_shares
                           FROM trading.positions WHERE account_id = %s ORDER BY symbol""", (account_id,))
            positions = [{"symbol": row["symbol"], "name": row["name"], "shares": row["shares"], "cost_basis": row["cost_basis"],
                          "acquired": row["acquired"], "recent_shares": row["recent_shares"]} for row in cur.fetchall()]
            held_cash, held_shares = self._reserved(cur, account_id, release)
            return Portfolio(user=account["display_name"], currency=account["currency"], cash_avail=account["cash_balance"] - held_cash, positions=positions, note=account["note"],
                             reserved_cash=held_cash, reserved_shares=held_shares)
        return self._run(account_id, fn)

    def available_cash(self, account_id: int) -> Decimal:
        """The cash the account may spend: its cash balance less the cash its open orders have set aside (design choices entry 140). `account_balance` is the whole balance."""
        def fn(cur):
            cur.execute("SELECT cash_balance FROM trading.accounts WHERE account_id = %s", (account_id,))
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"no account {account_id}")
            return row["cash_balance"] - self._reserved(cur, account_id)[0]
        return self._run(account_id, fn)

    def available_shares(self, account_id: int, symbol: str) -> Decimal:
        """The shares of `symbol` the account may sell: the shares held less the shares its open orders to sell have set aside (0 when none is held)."""
        symbol = check_symbol(symbol)
        def fn(cur):
            cur.execute("SELECT shares FROM trading.positions WHERE account_id = %s AND symbol = %s", (account_id, symbol))
            row = cur.fetchone()
            held = sh.as_decimal(row["shares"]) if row else Decimal(0)
            reserved = self._reserved(cur, account_id)[1].get(symbol, Decimal(0))
            return Decimal(0) if reserved >= held else sh.quantity(held - reserved)
        return self._run(account_id, fn)

    def history(self, account_id: int) -> dict:
        """The account's own dated record, READ ONLY (the Proving Ground's live monitor draws the account's equity from it): `trades` (every row of `trading.trades`, oldest first: symbol, side, shares, price, traded_on, record_id,
        source) and `cash` (every row of `trading.cash_ledger`, oldest first: kind, amount, balance_after, occurred_on). Nothing is written; a SELECT of the account's own rows."""
        def fn(cur):
            cur.execute("SELECT 1 FROM trading.accounts WHERE account_id = %s", (account_id,))
            if cur.fetchone() is None:
                raise ValueError(f"no account {account_id}")
            cur.execute("""SELECT trade_id, symbol, side, shares, price, traded_on, record_id, source FROM trading.trades
                           WHERE account_id = %s ORDER BY traded_on, trade_id""", (account_id,))
            trades = [dict(row) for row in cur.fetchall()]
            cur.execute("""SELECT entry_id, kind, amount, balance_after, occurred_on FROM trading.cash_ledger
                           WHERE account_id = %s ORDER BY occurred_on, entry_id""", (account_id,))
            return {"trades": trades, "cash": [dict(row) for row in cur.fetchall()]}
        return self._run(account_id, fn)

    def record_trade(self, account_id: int, symbol: str, side: str, shares: Decimal, price: Decimal, traded_on: date, *,
                     name: str | None = None, source: str = "bot", record_id: str | None = None) -> Decimal:
        """Book a filled trade. Returns that trade's realized gain (0 for a buy). Does not send an order to a broker. This is `book_trade` and its gain."""
        return self.book_trade(account_id, symbol, side, shares, price, traded_on, name=name, source=source, record_id=record_id).gain

    def book_trade(self, account_id: int, symbol: str, side: str, shares: Decimal, price: Decimal, traded_on: date, *,
                   name: str | None = None, source: str = "bot", record_id: str | None = None, order_id: int | None = None) -> "Booked":
        """Book a filled trade in ONE transaction and say what it did (`Booked`). The user's PORTFOLIO is locked first, with row-level locks held until the transaction ends: the account row (the cash) and EVERY position
        row of the account (FOR UPDATE, in symbol order so two bookings cannot deadlock), so no other booking can read or change the cash or a holding in between (a lock that is not had in `BOOKING_LOCK_SECONDS`
        fails the booking and nothing is committed). The account must be active, a purchase must be covered by the cash and a sale by the shares held; then the cash (through the ledger) and the position move together
        and the trade row is written, and BEFORE the transaction commits the account is read again inside it: the cash must be the cash before less the cost (or plus the proceeds), the shares of the symbol must be
        the shares before plus (or less) the trade, every other holding must be as it was, and the trade row must be there. Anything else raises `BookingCheckError` and the whole transaction is rolled back, so nothing
        is committed (design choices entry 82). A refusal of the book raises ValueError with its sentence; nothing changed. `shares` is a quantity above zero (`shares.quantity`: a whole number, or a number with up to six
        decimals, a Decimal; a bool and a string are refused) and `price` a Decimal of at most four places (`parse_price`: a float is refused); one with a fraction is refused, before anything is written, by a book whose share columns are still `integer` (`holds_fractions`; design choices entry 110).

        WAITING ORDERS (design choices entry 140). The account's open orders hold cash and shares, and the lock this transaction takes covers them too (they are read under the same account-row lock every placement, fill, expiry and cancel takes first): a purchase must
        be covered by the cash LESS what the other open orders hold, a sale by the shares held LESS what the other open orders to sell hold, so a booking can never use what a waiting order has set aside. `order_id` is the open order this trade FILLS: its own hold is not
        counted against it (it must be `open`, for this symbol and side, and the trade may not cost more cash or sell more shares than it holds), and in the SAME transaction it becomes `filled` with this trade's row and its hold is gone. A second fill of the
        same order finds it no longer open and is refused, so a fill can never book twice. Before the commit the order's row and the other orders' holds are read again with the rest."""
        symbol = check_symbol(symbol)
        if side not in ("buy", "sell"):
            raise ValueError("side must be buy or sell")
        shares = self._count_of(shares)
        price = parse_price(price)
        if type(traded_on) is not date:
            raise ValueError("traded_on must be a date")
        if source not in ("bot", "user"):
            raise ValueError("source must be bot or user")
        if record_id is not None and (not isinstance(record_id, str) or not record_id.strip() or len(record_id) > 80):
            raise ValueError("record_id must be text of at most 80 characters")
        held_name = None if name is None else clean_optional_name(name, limit=200)

        def fn(cur):
            cur.execute("SELECT set_config('lock_timeout', %s, true)", (f"{BOOKING_LOCK_SECONDS}s",))       # for this transaction only: a lock that is not had in time fails the booking
            account = self._lock(cur, account_id)                                                      # the account row (the cash): FOR UPDATE
            _require_active(account)
            cur.execute("""SELECT symbol, name, shares, cost_basis, acquired, recent_shares
                           FROM trading.positions WHERE account_id = %s ORDER BY symbol FOR UPDATE""", (account_id,))   # EVERY position row of the portfolio: FOR UPDATE, until the commit
            holdings = {row["symbol"]: row for row in cur.fetchall()}
            position = holdings.get(symbol)
            others_before = {name: sh.as_decimal(row["shares"]) for name, row in holdings.items() if name != symbol}
            cash = account["cash_balance"]
            held = sh.as_decimal(position["shares"]) if position else Decimal(0)
            others_cash, others_shares = self._reserved(cur, account_id, release=order_id, lock=True)      # what the OTHER open orders hold: it stays covered
            filled = self._order_to_fill(cur, account_id, order_id, symbol, side) if order_id is not None else None
            if side == "buy":
                cost = cents(price * shares)
                if cost <= 0:
                    raise ValueError("the trade must move at least one cent")
                if filled is not None and filled.get("raise_funds_by") is None and cost > filled["reserved_cash"]:      # an order the user permitted to raise funds (entry 143) set aside only the cash that was free: the money for the rest came from the sales booked before this purchase, and the cash check below still holds
                    raise ValueError(f"the fill costs ${cost:,.2f} and the order set aside only ${filled['reserved_cash']:,.2f}: a fill never costs more than was set aside")
                if cash - others_cash < cost:
                    raise ValueError(f"cash ${cash:,.2f} does not cover ${cost:,.2f}" if not others_cash else
                                     f"cash ${cash:,.2f}, less ${others_cash:,.2f} set aside for waiting orders, does not cover ${cost:,.2f}")
                _buy_position(cur, account_id, symbol, shares, price, traded_on, held_name, position)
                gain = Decimal("0.00")
                self._ledger(cur, account_id, "buy", -cost, cents(cash - cost), traded_on, f"buy {sh.text(shares)} {symbol}")
                want_cash, want_held = cents(cash - cost), held + shares
            else:
                set_aside = others_shares.get(symbol, Decimal(0))
                if filled is not None and shares > filled["reserved_shares"]:
                    raise ValueError(f"the fill sells {sh.text(shares)} and the order set aside only {sh.text(filled['reserved_shares'])}: a fill never sells more than was set aside")
                if position is None or shares > held or shares > held - set_aside:
                    raise ValueError(f"cannot sell {sh.text(shares)} of {symbol}: the portfolio holds {sh.text(held)}" if not set_aside else
                                     f"cannot sell {sh.text(shares)} of {symbol}: the portfolio holds {sh.text(held)} and {sh.text(set_aside)} of them are set aside for waiting orders")
                proceeds = cents(price * shares)
                if proceeds <= 0:
                    raise ValueError("the trade must move at least one cent")
                gain = proceeds - cents(position["cost_basis"] * shares)
                _sell_position(cur, account_id, symbol, shares, position)
                self._ledger(cur, account_id, "sell", proceeds, cents(cash + proceeds), traded_on, f"sell {sh.text(shares)} {symbol}")
                want_cash, want_held = cents(cash + proceeds), held - shares
            cur.execute("""INSERT INTO trading.trades (account_id, symbol, side, shares, price, realized_gain, traded_on, record_id, source)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING trade_id""",
                        (account_id, symbol, side, shares, price, gain, traded_on, None if record_id is None else record_id.strip(), source))
            row = cur.fetchone()["trade_id"]
            if order_id is not None:      # the order is filled by THIS trade, in this transaction: its hold is gone with its `open` status (a second fill finds it filled and is refused above)
                cur.execute("""UPDATE trading.open_orders SET status = 'filled', closed_at = now(), trade_id = %s, reason = %s
                               WHERE order_id = %s AND account_id = %s AND status = 'open'""",
                            (row, f"filled: {side} {sh.text(shares)} {symbol} at {price:,.4f}", order_id, account_id))
                if cur.rowcount != 1:
                    raise BookingCheckError(f"the open order {order_id} was not marked filled: the booking was rolled back and nothing was committed")
            # the check, inside the transaction and before it commits: read the account again and compare it with what the trade must have done
            now_cash = self._cash(cur, account_id)
            cur.execute("SELECT symbol, shares FROM trading.positions WHERE account_id = %s", (account_id,))
            now_all = {r["symbol"]: sh.as_decimal(r["shares"]) for r in cur.fetchall()}
            now_held = sh.as_decimal(now_all[symbol]) if symbol in now_all else Decimal(0)                       # compared at scale 6, exactly
            others_after = {name: n for name, n in now_all.items() if name != symbol}
            cur.execute("SELECT 1 FROM trading.trades WHERE trade_id = %s AND account_id = %s AND symbol = %s AND side = %s AND shares = %s", (row, account_id, symbol, side, shares))
            booked_trade = cur.fetchone() is not None
            after_cash, after_shares = self._reserved(cur, account_id)      # the other orders' holds are exactly what they were, and the order this trade filled holds nothing now
            order_marked = True
            if order_id is not None:
                cur.execute("SELECT 1 FROM trading.open_orders WHERE order_id = %s AND account_id = %s AND status = 'filled' AND trade_id = %s", (order_id, account_id, row))
                order_marked = cur.fetchone() is not None
            if now_cash != want_cash or now_held != want_held or others_after != others_before or not booked_trade or not order_marked or after_cash != others_cash or after_shares != others_shares:
                raise BookingCheckError(f"the account did not show the {side} of {sh.text(shares)} {symbol} when it was read again before the commit (cash ${now_cash:,.2f}, expected ${want_cash:,.2f}; "
                                        f"{sh.text(now_held)} shares, expected {sh.text(want_held)}{'' if others_after == others_before else '; another holding changed'}): the booking was rolled back and nothing was committed")
            return Booked(trade_row=row, gain=gain, symbol=symbol, side=side, shares=sh.quantity(shares), price=price, cash_before=cash, cash_after=now_cash,
                          held_before=sh.quantity(held), held_after=sh.quantity(now_held), order_id=order_id)
        return self._run(account_id, fn)

    def _count_of(self, shares) -> Decimal:
        """The shares of a trade as the Decimal at scale 6 the book works with. Raises ValueError, before anything is written, for a number that is not above zero or has more than six decimals, and for a quantity with
        a fraction on a book whose share columns are still integer (`holds_fractions`): nothing is ever rounded into an integer column."""
        try:
            quantity = sh.quantity(shares)
        except ValueError:
            raise ValueError("shares must be a number above zero (a whole number, or one with at most 6 decimals)") from None
        if quantity <= 0:
            raise ValueError("shares must be a number above zero (a whole number, or one with at most 6 decimals)")
        if not sh.whole(quantity) and not self.holds_fractions():
            raise ValueError(FRACTION_REFUSED)
        return sh.as_decimal(quantity)

    # ---- waiting orders (design choices entry 140): the table `trading.open_orders`, the reservations, the placement, the expiry and the cancel

    def _has_orders_table(self, cur) -> bool:
        """Whether `trading.open_orders` exists (a True is remembered: a table is never dropped). A book that `bash scripts/setup_postgres.sh` has not been run on since entry 140 has no such table, and no order holds anything."""
        if self._orders_table:
            return True
        cur.execute("SELECT to_regclass('trading.open_orders') IS NOT NULL AS there")
        self._orders_table = bool(cur.fetchone()["there"])
        return self._orders_table

    def _reserved(self, cur, account_id: int, release: int | None = None, lock: bool = False) -> tuple[Decimal, dict[str, Decimal]]:
        """What the account's OPEN orders hold, read inside the caller's transaction: (the cash, {symbol: shares}). The ONE place a reservation is added up, so that every reader of what is available adds it up the same way. `release` leaves one
        order out (the one being filled). `lock` takes a row-level lock on the open orders (FOR UPDATE, in order number) for the rest of the transaction: what a placement, a booking and a withdrawal do after the account row is locked."""
        if not self._has_orders_table(cur):
            return Decimal("0.00"), {}
        cur.execute(f"""SELECT order_id, symbol, reserved_cash, reserved_shares FROM trading.open_orders
                        WHERE account_id = %s AND status = 'open' AND order_id IS DISTINCT FROM %s ORDER BY order_id{' FOR UPDATE' if lock else ''}""", (account_id, release))
        cash, shares = Decimal("0.00"), {}
        for row in cur.fetchall():
            cash += row["reserved_cash"]
            if row["reserved_shares"] > 0:
                shares[row["symbol"]] = sh.quantity(shares.get(row["symbol"], Decimal(0)) + row["reserved_shares"])
        return cents(cash), shares

    def _order_to_fill(self, cur, account_id: int, order_id: int, symbol: str, side: str) -> dict:
        """The open order a booking fills, locked (FOR UPDATE). Raises ValueError, with the reason, when there is no such order of this account, it is not `open` any more (so it can never be booked twice), or it is for another symbol or side."""
        cur.execute("""SELECT * FROM trading.open_orders WHERE order_id = %s AND account_id = %s FOR UPDATE""", (order_id, account_id))      # all the columns: a book without `raise_funds_by` (entry 143) has none to read and `.get` says None
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"there is no open order {order_id} on this account")
        if row["status"] != "open":
            raise ValueError(f"the order {order_id} is {row['status']}, not open: it cannot be filled")
        if (row["symbol"], row["side"]) != (symbol, side):
            raise ValueError(f"the order {order_id} is to {row['side']} {row['symbol']}, not to {side} {symbol}")
        return row

    def place_open_order(self, account_id: int, *, request_id: str, record_id: str, symbol: str, side: str, limit_price, request: str, placed_at: datetime, expires_at: datetime,
                         shares=None, dollars=None, company: str | None = None, raise_funds_by: str | None = None) -> "OpenOrder":
        """Place a waiting DAY order and SET ASIDE what it needs, in ONE transaction (design choices entry 140). `raise_funds_by` (entry 143: "words" or "setting", a purchase only) says the user permitted selling holdings to pay for the purchase when
        its price is reached: such an order sets aside the cash that is FREE (up to its most possible cost, and it may be none) instead of being refused for the rest, and nothing is sold while it waits. The account row is locked first (the lock every booking, expiry and cancel takes), then EVERY position row and the account's open
        orders, so a placement cannot race a fill or another placement. A purchase sets aside its most possible cost: `shares` times the limit price to the cent, or its `dollars` (an order in dollars); a sale sets aside its `shares`. It is refused
        (ValueError, nothing written) when the account is not active, when the cash available (the cash less what open orders already hold) does not cover a purchase, or when the shares not already set aside do not cover a sale: the book is the guard,
        whatever a caller read before. `limit_price` is the ceiling of a purchase and the floor of a sale (a Decimal, `parse_price`); exactly one of `shares` (`shares.quantity`) and `dollars` (`parse_money`, a purchase only) is given; `placed_at` and `expires_at`
        are aware moments. Returns the `OpenOrder`."""
        symbol, record_id, request_id = check_symbol(symbol), _text(record_id, "record_id", 80), _text(request_id, "request_id", 80)
        if side not in ("buy", "sell"):
            raise ValueError("side must be buy or sell")
        limit = parse_price(limit_price)
        if (shares is None) == (dollars is None):
            raise ValueError("a waiting order has shares or dollars, one of the two")
        if dollars is not None and side != "buy":
            raise ValueError("only a purchase can be an order in dollars")
        count = self._count_of(shares) if shares is not None else None
        money = parse_money(dollars) if dollars is not None else None
        if money is not None and money <= 0:
            raise ValueError("the dollars of a waiting order must be above zero")
        for moment, label in ((placed_at, "placed_at"), (expires_at, "expires_at")):
            if not isinstance(moment, datetime) or moment.tzinfo is None or moment.utcoffset() is None:
                raise ValueError(f"{label} must be a datetime with a timezone")
        if expires_at <= placed_at:
            raise ValueError("a waiting order must expire after it is placed")
        text = _text(request, "request", 2000)
        if raise_funds_by is not None and (side != "buy" or raise_funds_by not in ("words", "setting")):
            raise ValueError("raise_funds_by is words or setting, and only for a purchase")
        held_name = None if company is None else clean_optional_name(company, limit=200)
        need_cash = (trade_amount(count, limit) if count is not None else money) if side == "buy" else Decimal("0.00")
        need_shares = count if side == "sell" else Decimal(0)
        if side == "buy" and need_cash <= 0:
            raise ValueError("the order must set aside at least one cent")

        def fn(cur):
            cur.execute("SELECT set_config('lock_timeout', %s, true)", (f"{BOOKING_LOCK_SECONDS}s",))
            account = self._lock(cur, account_id)
            _require_active(account)
            if not self._has_orders_table(cur):
                raise ValueError("this account book has no table for waiting orders: apply the new schema with `bash scripts/setup_postgres.sh` (it keeps every table and account)")
            cur.execute("SELECT symbol, shares FROM trading.positions WHERE account_id = %s ORDER BY symbol FOR UPDATE", (account_id,))
            held = {row["symbol"]: sh.as_decimal(row["shares"]) for row in cur.fetchall()}
            set_cash, set_shares = self._reserved(cur, account_id, lock=True)
            hold = need_cash
            if side == "buy":
                free = account["cash_balance"] - set_cash
                if hold > free and raise_funds_by is not None:      # entry 143: the free cash is set aside, the rest is raised by selling when the price is reached
                    hold = max(free, Decimal("0.00"))
                if hold > free:
                    raise ValueError(f"the account has ${account['cash_balance']:,.2f} in cash" + (f", ${set_cash:,.2f} of it set aside for other waiting orders," if set_cash else "") +
                                     f" so ${free:,.2f} is available, and this order would set aside ${hold:,.2f}")
            else:
                free = held.get(symbol, Decimal(0)) - set_shares.get(symbol, Decimal(0))
                if need_shares > free:
                    raise ValueError(f"the account holds {sh.text(held.get(symbol, Decimal(0)))} of {symbol}" + (f", {sh.text(set_shares[symbol])} of them set aside for other waiting orders," if set_shares.get(symbol) else "") +
                                     f" so {sh.text(max(free, Decimal(0)))} can be sold, and this order would set aside {sh.text(need_shares)}")
            extra = ("raise_funds_by", ", %s") if raise_funds_by is not None else ("", "")      # the column is written only when it is used: a book that has not had the schema of entry 143 still takes every other order
            cur.execute(f"""INSERT INTO trading.open_orders (account_id, request_id, record_id, symbol, company, side, shares, dollars, limit_price, reserved_cash, reserved_shares, request, placed_at, expires_at{', ' + extra[0] if extra[0] else ''})
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s{extra[1]}) RETURNING *""",
                        (account_id, request_id, record_id, symbol, held_name, side, count, money, limit, hold, sh.as_decimal(need_shares), text, placed_at, expires_at, *([raise_funds_by] if raise_funds_by is not None else [])))
            return OpenOrder.from_row(cur.fetchone())
        return self._run(account_id, fn)

    def open_orders(self, account_id: int, statuses: tuple[str, ...] = ("open",)) -> list["OpenOrder"]:
        """The account's orders with one of these statuses, in order number (the waiting ones by default)."""
        bad = [s for s in statuses if s not in OPEN_ORDER_STATUSES]
        if bad:
            raise ValueError(f"an order's status is one of {', '.join(OPEN_ORDER_STATUSES)}, not {bad[0]!r}")
        def fn(cur):
            if not self._has_orders_table(cur):
                return []
            cur.execute("SELECT * FROM trading.open_orders WHERE account_id = %s AND status = ANY(%s) ORDER BY order_id", (account_id, list(statuses)))
            return [OpenOrder.from_row(row) for row in cur.fetchall()]
        return self._run(account_id, fn)

    def get_open_order(self, account_id: int, order_id: int) -> "OpenOrder | None":
        """One order of the account, whatever its status, or None."""
        def fn(cur):
            if not self._has_orders_table(cur):
                return None
            cur.execute("SELECT * FROM trading.open_orders WHERE account_id = %s AND order_id = %s", (account_id, order_id))
            row = cur.fetchone()
            return None if row is None else OpenOrder.from_row(row)
        return self._run(account_id, fn)

    def accounts_with_open_orders(self) -> list[int]:
        """The numbers of the accounts that have an open order (the watcher's first question: row security hides every other account's rows from this role, and this answer is an id and nothing else)."""
        def fn(cur):
            if not self._has_orders_table(cur):
                return []
            cur.execute("SELECT account_id FROM trading.accounts_with_open_orders()")
            return [row["account_id"] for row in cur.fetchall()]
        return self._run(None, fn)

    def account_of_open_order(self, order_id: int) -> int | None:
        """The account an order belongs to (a cancel by order number needs it), or None when there is no such order."""
        def fn(cur):
            if not self._has_orders_table(cur):
                return None
            cur.execute("SELECT trading.account_of_open_order(%s) AS account_id", (order_id,))
            return cur.fetchone()["account_id"]
        return self._run(None, fn)

    def _close_orders(self, account_id: int, status: str, reason: str, when: datetime | None, order_id: int | None = None, expired_by: datetime | None = None) -> list["OpenOrder"]:
        """Close open orders in ONE transaction (the account row is locked first, as every booking does, so a close cannot race a fill): the order `order_id`, or every order whose `expires_at` is at or before `expired_by`. What they held is released
        with their `open` status. Returns the orders closed; never one that was not `open`."""
        when = when or datetime.now(timezone.utc)
        if not isinstance(when, datetime) or when.tzinfo is None or when.utcoffset() is None:
            raise ValueError("the moment must be a datetime with a timezone")
        def fn(cur):
            self._lock(cur, account_id)
            if not self._has_orders_table(cur):
                return []
            cur.execute("""UPDATE trading.open_orders SET status = %s, closed_at = %s, reason = %s
                           WHERE account_id = %s AND status = 'open' AND (%s::bigint IS NULL OR order_id = %s) AND (%s::timestamptz IS NULL OR expires_at <= %s) RETURNING *""",
                        (status, when, " ".join(reason.split())[:500], account_id, order_id, order_id, expired_by, expired_by))
            return [OpenOrder.from_row(row) for row in sorted(cur.fetchall(), key=lambda r: r["order_id"])]
        return self._run(account_id, fn)

    def cancel_open_order(self, account_id: int, order_id: int, when: datetime | None = None, reason: str = "cancelled by the user") -> "OpenOrder":
        """The user cancels a waiting order: it becomes `cancelled` and what it set aside is available again, in one transaction. ValueError (nothing changed) when there is no such order or it is not `open` (filled, expired, cancelled or rejected)."""
        closed = self._close_orders(account_id, "cancelled", reason, when, order_id=order_id)
        if not closed:
            found = self.get_open_order(account_id, order_id)
            raise ValueError(f"there is no order {order_id} on this account" if found is None else f"the order {order_id} is {found.status}: it cannot be cancelled")
        return closed[0]

    def reject_open_order(self, account_id: int, order_id: int, reason: str, when: datetime | None = None) -> "OpenOrder | None":
        """A fill that failed (the book refused it, it was rolled back, its booking could not be confirmed): the order becomes `rejected` with the reason, is never retried, and what it set aside is available again (entry 82). None when it is not `open` any more."""
        closed = self._close_orders(account_id, "rejected", reason, when, order_id=order_id)
        return closed[0] if closed else None

    def expire_open_orders(self, account_id: int, now: datetime) -> list["OpenOrder"]:
        """A DAY order does not outlive its session: every open order of the account whose `expires_at` is at or before `now` becomes `expired` and what it set aside is available again, in one transaction. Returns the orders expired."""
        return self._close_orders(account_id, "expired", "expired at the close of its session: a day order is never carried to another day", now, expired_by=now)

    def watcher_lock(self) -> "WatcherLock":
        """The Postgres advisory lock of the watcher (one watcher for each database)."""
        return WatcherLock(self._connect)

    def watcher_running(self) -> bool:
        """Whether a watcher holds the lock now (a server is running its watcher): the lock is tried on a connection of its own and let go at once. A database that cannot be reached counts as no watcher."""
        probe = self.watcher_lock()
        try:
            got = probe.try_acquire()
        except Exception:
            return False
        finally:
            probe.release()
        return not got

    def read_booked(self, account_id: int, trade_row: int) -> dict | None:
        """Read one booked trade back from the book (a NEW connection and transaction: what the commit made durable): its row of `trading.trades` as a dict, or None when there is no such row. The Account Manager
        calls it right after `book_trade` committed, to confirm that the order was applied (design choices entry 82)."""
        def fn(cur):
            cur.execute("""SELECT trade_id, symbol, side, shares, price, traded_on, record_id, source FROM trading.trades
                           WHERE account_id = %s AND trade_id = %s""", (account_id, trade_row))
            return cur.fetchone()
        return self._run(account_id, fn)

    def calculate_total_gain(self, account_id: int, start, end) -> Decimal:
        start, end = date_window(start, end)
        def fn(cur):
            cur.execute("""SELECT COALESCE(SUM(realized_gain), 0::numeric) AS gain FROM trading.trades
                           WHERE account_id = %s AND traded_on BETWEEN %s AND %s""", (account_id, start, end))
            return cur.fetchone()["gain"]
        return self._run(account_id, fn)

    def number_trades(self, account_id: int, start, end) -> int:
        return self._count(account_id, start, end, "SELECT count(*) AS n FROM trading.trades WHERE account_id = %s AND traded_on BETWEEN %s AND %s")

    def number_approved_recommendations(self, account_id: int, start, end) -> int:
        return self._count(account_id, start, end, """SELECT count(*) AS n FROM trading.recommendation_decisions
                           WHERE account_id = %s AND decided_on BETWEEN %s AND %s AND will_trade = ANY(%s)""", APPROVED_DECISIONS)

    def number_denied_recommendations(self, account_id: int, start, end) -> int:
        return self._count(account_id, start, end, """SELECT count(*) AS n FROM trading.recommendation_decisions
                           WHERE account_id = %s AND decided_on BETWEEN %s AND %s AND will_trade = ANY(%s)""", DENIED_DECISIONS)

    def save_decisions(self, account_id: int, decisions: list[dict]) -> None:
        rows = [_decision_row(account_id, item) for item in decisions]
        if not rows:
            return
        if any(not sh.whole(row[4]) for row in rows) and not self.holds_fractions():
            raise ValueError(FRACTION_REFUSED)

        def fn(cur):
            self._lock(cur, account_id)
            cur.executemany("""INSERT INTO trading.recommendation_decisions
                                 (account_id, record_id, symbol, action, shares, mode, will_trade, reason, decided_on)
                               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                               ON CONFLICT (account_id, record_id) DO UPDATE SET
                                 symbol = EXCLUDED.symbol, action = EXCLUDED.action, shares = EXCLUDED.shares,
                                 mode = EXCLUDED.mode, will_trade = EXCLUDED.will_trade, reason = EXCLUDED.reason,
                                 decided_on = EXCLUDED.decided_on""", rows)
        self._run(account_id, fn)

    def _count(self, account_id: int, start, end, sql: str, states: tuple[str, ...] | None = None) -> int:
        start, end = date_window(start, end)
        params = (account_id, start, end) if states is None else (account_id, start, end, list(states))
        def fn(cur):
            cur.execute(sql, params)
            return int(cur.fetchone()["n"])
        return self._run(account_id, fn)

    def _account_row(self, account_id: int) -> dict:
        def fn(cur):
            cur.execute("SELECT * FROM trading.accounts WHERE account_id = %s", (account_id,))
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"no account {account_id}")
            return row
        return self._run(account_id, fn)

    def _connection(self):
        return psycopg.connect(row_factory=dict_row, **self._connect)

    def _run(self, account_id: int | None, fn):
        try:
            with self._connection() as conn:
                with conn.cursor() as cur:
                    if account_id is not None:
                        cur.execute("SELECT set_config('trading.account_id', %s, true)", (str(account_id),))
                    return fn(cur)
        except (errors.RaiseException, errors.CheckViolation) as error:
            raise ValueError(_db_message(error)) from None

    def _lock(self, cur, account_id: int) -> dict:
        cur.execute("SELECT account_id, display_name, status, cash_balance FROM trading.accounts WHERE account_id = %s FOR UPDATE", (account_id,))
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"no account {account_id}")
        return row

    def _cash(self, cur, account_id: int) -> Decimal:
        cur.execute("SELECT cash_balance FROM trading.accounts WHERE account_id = %s", (account_id,))
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"no account {account_id}")
        return row["cash_balance"]

    def _ledger(self, cur, account_id: int, kind: str, amount: Decimal, balance_after: Decimal, occurred_on: date, memo: str) -> None:
        cur.execute("""INSERT INTO trading.cash_ledger (account_id, kind, amount, balance_after, memo, occurred_on)
                       VALUES (%s, %s, %s, %s, %s, %s)""", (account_id, kind, amount, balance_after, memo, occurred_on))


class BookingCheckError(ValueError):
    """The trade was written and the account did not show it when it was read again INSIDE the same transaction, so the transaction was rolled back: nothing was committed (design choices entry 82)."""


@dataclass(frozen=True)
class Booked:
    """What one committed booking did, as the transaction saw it while it held the account row (FOR UPDATE, so no other booking can be in between): the row of the trade (`trade_row`, `trading.trades.trade_id`),
    its realized gain, and the cash and the shares of the symbol before and after. The Account Manager checks these against a read from a NEW connection once the transaction has committed (`read_booked`)."""
    trade_row: int
    gain: Decimal
    symbol: str
    side: str
    shares: Decimal                     # quantities: Decimals in their canonical form (a whole number has no decimals; entries 110 and 112)
    price: Decimal
    cash_before: Decimal
    cash_after: Decimal
    held_before: Decimal
    held_after: Decimal
    order_id: int | None = None         # the open order this trade filled (design choices entry 140), or None


@dataclass(frozen=True)
class OpenOrder:
    """A waiting DAY order as `trading.open_orders` holds it (design choices entry 140). `limit_price` is the ceiling of a purchase and the floor of a sale; `shares` or `dollars` (an order in dollars); while `status` is `open` it holds
    `reserved_cash` (a purchase) or `reserved_shares` (a sale); `trade_id` is the booked trade once it is `filled`."""
    order_id: int
    account_id: int
    request_id: str
    record_id: str
    symbol: str
    company: str | None
    side: str
    shares: Decimal | None
    dollars: Decimal | None
    limit_price: Decimal
    reserved_cash: Decimal
    reserved_shares: Decimal
    request: str
    placed_at: datetime
    expires_at: datetime
    status: str
    closed_at: datetime | None
    reason: str | None
    trade_id: int | None
    raise_funds_by: str | None = None      # entry 143: "words" or "setting" when the user permitted selling holdings to pay for this purchase when its price is reached (None: never)

    @classmethod
    def from_row(cls, row: dict) -> "OpenOrder":
        return cls(order_id=row["order_id"], account_id=row["account_id"], request_id=row["request_id"], record_id=row["record_id"], symbol=row["symbol"], company=row["company"], side=row["side"],
                   shares=None if row["shares"] is None else sh.quantity(row["shares"]), dollars=row["dollars"], limit_price=row["limit_price"], reserved_cash=row["reserved_cash"],
                   reserved_shares=sh.quantity(row["reserved_shares"]), request=row["request"], placed_at=row["placed_at"], expires_at=row["expires_at"], status=row["status"], closed_at=row["closed_at"],
                   reason=row["reason"], trade_id=row["trade_id"], raise_funds_by=row.get("raise_funds_by"))

    def as_dict(self) -> dict:
        """The order as the page and the reports show it: plain values (Decimals stay Decimals: the one encoder writes them as numbers; moments are UTC text)."""
        utc = lambda moment: None if moment is None else moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return {"order_id": self.order_id, "account_id": self.account_id, "request_id": self.request_id, "record_id": self.record_id, "symbol": self.symbol, "company": self.company, "side": self.side,
                "shares": self.shares, "dollars": self.dollars, "limit_price": self.limit_price, "reserved_cash": self.reserved_cash, "reserved_shares": self.reserved_shares, "request": self.request,
                "placed_utc": utc(self.placed_at), "expires_utc": utc(self.expires_at), "status": self.status, "closed_utc": utc(self.closed_at), "reason": self.reason, "trade_id": self.trade_id, "raise_funds_by": self.raise_funds_by}


class WatcherLock:
    """The Postgres ADVISORY LOCK that makes one watcher of the waiting orders for each database (design choices entry 140, W5): held on a connection of its own for as long as the watcher lives, so that a second server does not check the
    same orders. `try_acquire` is True when this holds the lock (also when it already did, and the connection is still there), False when another connection does; `release` lets go. The lock is the connection's: when the connection
    dies Postgres frees it, and the next `try_acquire` takes it again if nobody else has."""

    def __init__(self, connect: dict):
        self._connect, self._conn = connect, None

    def try_acquire(self) -> bool:
        if self._conn is not None:
            try:
                self._conn.execute("SELECT 1")      # still alive: still held
                return True
            except Exception:
                self.release()
        conn = psycopg.connect(row_factory=dict_row, autocommit=True, **self._connect)
        try:
            got = bool(conn.execute("SELECT pg_try_advisory_lock(%s) AS got", (WATCHER_LOCK_KEY,)).fetchone()["got"])
        except Exception:
            conn.close()
            raise
        if got:
            self._conn = conn
        else:
            conn.close()
        return got

    def release(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()      # closing the connection lets the session's advisory lock go
            except Exception:
                pass


def _require_active(row: dict) -> None:
    if row["status"] != "active":
        raise ValueError(f"the account is {row['status']}: it cannot take funds or a trade")


def _buy_position(cur, account_id: int, symbol: str, shares: Decimal, price: Decimal, traded_on: date, name: str | None, position: dict | None) -> None:
    if position is None:
        cur.execute("""INSERT INTO trading.positions (account_id, symbol, name, shares, cost_basis, acquired, recent_shares)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)""", (account_id, symbol, name, shares, price, traded_on, shares))
        return
    before = sh.as_decimal(position["shares"])
    total = before + shares
    basis = price_scale(EXACT.divide(EXACT.add(EXACT.multiply(before, position["cost_basis"]), EXACT.multiply(shares, price)), total))
    if position["acquired"] is not None and position["acquired"] >= traded_on:
        locked = None if position["recent_shares"] is None else sh.as_decimal(position["recent_shares"]) + shares
    else:
        locked = shares
    cur.execute("""UPDATE trading.positions
                   SET name = COALESCE(%s, name), shares = %s, cost_basis = %s, acquired = %s, recent_shares = %s
                   WHERE account_id = %s AND symbol = %s""", (name, total, basis, traded_on, locked, account_id, symbol))


def _sell_position(cur, account_id: int, symbol: str, shares: Decimal, position: dict) -> None:
    left = sh.as_decimal(position["shares"]) - shares
    if left == 0:                                                              # exactly 0 is left: the position row is gone
        cur.execute("DELETE FROM trading.positions WHERE account_id = %s AND symbol = %s", (account_id, symbol))
        return
    recent = None if position["recent_shares"] is None else sh.as_decimal(position["recent_shares"])
    if recent is not None:
        recent = min(recent, left) or None
    cur.execute("UPDATE trading.positions SET shares = %s, recent_shares = %s WHERE account_id = %s AND symbol = %s", (left, recent, account_id, symbol))


def _decision_row(account_id: int, decision: dict) -> tuple:
    will_trade = decision.get("will_trade")
    if will_trade not in DECISION_STATES:
        raise ValueError(f"will_trade must be one of {', '.join(DECISION_STATES)}")
    record_id = decision.get("record_id")
    if not isinstance(record_id, str) or not record_id.strip() or len(record_id.strip()) > 80:
        raise ValueError("a decision needs a record_id")
    text = decision.get("decided_utc")
    if not isinstance(text, str) or len(text) < 10:
        raise ValueError("a decision needs decided_utc")
    try:
        decided_on = date.fromisoformat(text[:10])
    except ValueError:
        raise ValueError("a decision needs decided_utc") from None
    action = decision.get("action")
    if not isinstance(action, str) or not action.isidentifier() or len(action) > 32:
        raise ValueError("a decision needs an action such as add or reduce")
    try:
        shares = sh.quantity(decision.get("shares"))                           # zero or more: a decision that sized to nothing is saved with 0
    except ValueError:
        raise ValueError("a decision needs a share count (a number of shares, zero or more, with at most six decimals)") from None
    mode = decision.get("mode")
    if mode not in ("ask", "auto"):
        raise ValueError("a decision needs mode ask or auto")
    reason = decision.get("reason")
    if not isinstance(reason, str):
        raise ValueError("a decision needs a reason")
    return (account_id, record_id.strip(), check_symbol(decision.get("symbol") or ""), action, sh.as_decimal(shares), mode, will_trade, reason, decided_on)


def _user(row: dict) -> UserAccount:
    horizon = row["investment_horizon_years"]
    return UserAccount(
        account_id=row["account_id"], display_name=row["display_name"], preferred_name=row["preferred_name"], email=row["email"],
        phone=row["phone"], timezone=row["timezone"], currency=row["currency"], status=row["status"], risk_tolerance=row["risk_tolerance"],
        trade_mode=row["trade_mode"], age=row["age"], investment_horizon_years=None if horizon is None else float(horizon), note=row["note"],
        seed_key=row["seed_key"], created_at=row["created_at"], updated_at=row["updated_at"])


def _text(value, label: str, limit: int) -> str:
    """Text that is not blank and at most `limit` characters (stripped). Raises ValueError, naming the field."""
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > limit:
        raise ValueError(f"{label} must be text of at most {limit} characters")
    return value.strip()


def _db_message(error: Exception) -> str:
    diag = getattr(error, "diag", None)
    text = getattr(diag, "message_primary", None) or str(error)
    return text.splitlines()[0]
