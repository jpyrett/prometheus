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
"""
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

import psycopg
from psycopg import errors
from psycopg.rows import dict_row

from prometheus.common.portfolio import Portfolio
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
BOOKING_LOCK_SECONDS = 10      # how long a booking waits for the locks on the account's portfolio before it fails (nothing is committed); a stuck lock must not hang a run
APPROVED_DECISIONS = ("yes_auto", "yes_user")
DENIED_DECISIONS = ("no_system_denied", "no_user_denied", "no_expired", "no_funds")
PENDING_DECISIONS = ("pending_user",)
DECISION_STATES = APPROVED_DECISIONS + DENIED_DECISIONS + PENDING_DECISIONS
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

    def __repr__(self) -> str:
        return f"AccountStore(host={self.config['host']!r}, database={self.config['database']!r}, user={self.config['app_user']!r})"

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
            trade_mode=row["trade_mode"], auto_min_score=row["auto_min_score"],
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

    def account_balance(self, account_id: int) -> Decimal:
        def fn(cur):
            return self._cash(cur, account_id)
        return self._run(account_id, fn)

    def list_portfolio(self, account_id: int) -> Portfolio:
        def fn(cur):
            cur.execute("SELECT display_name, currency, cash_balance, note FROM trading.accounts WHERE account_id = %s", (account_id,))
            account = cur.fetchone()
            if account is None:
                raise ValueError(f"no account {account_id}")
            cur.execute("""SELECT symbol, name, shares, cost_basis, acquired, recent_shares
                           FROM trading.positions WHERE account_id = %s ORDER BY symbol""", (account_id,))
            positions = [{"symbol": row["symbol"], "name": row["name"], "shares": row["shares"], "cost_basis": row["cost_basis"],
                          "acquired": row["acquired"], "recent_shares": row["recent_shares"]} for row in cur.fetchall()]
            return Portfolio(user=account["display_name"], currency=account["currency"], cash_avail=account["cash_balance"], positions=positions, note=account["note"])
        return self._run(account_id, fn)

    def record_trade(self, account_id: int, symbol: str, side: str, shares: int, price, traded_on: date, *,
                     name: str | None = None, source: str = "bot", record_id: str | None = None) -> Decimal:
        """Book a filled trade. Returns that trade's realized gain (0 for a buy). Does not send an order to a broker. This is `book_trade` and its gain."""
        return self.book_trade(account_id, symbol, side, shares, price, traded_on, name=name, source=source, record_id=record_id).gain

    def book_trade(self, account_id: int, symbol: str, side: str, shares: int, price, traded_on: date, *,
                   name: str | None = None, source: str = "bot", record_id: str | None = None) -> "Booked":
        """Book a filled trade in ONE transaction and say what it did (`Booked`). The user's PORTFOLIO is locked first, with row-level locks held until the transaction ends: the account row (the cash) and EVERY position
        row of the account (FOR UPDATE, in symbol order so two bookings cannot deadlock), so no other booking can read or change the cash or a holding in between (a lock that is not had in `BOOKING_LOCK_SECONDS`
        fails the booking and nothing is committed). The account must be active, a purchase must be covered by the cash and a sale by the shares held; then the cash (through the ledger) and the position move together
        and the trade row is written, and BEFORE the transaction commits the account is read again inside it: the cash must be the cash before less the cost (or plus the proceeds), the shares of the symbol must be
        the shares before plus (or less) the trade, every other holding must be as it was, and the trade row must be there. Anything else raises `BookingCheckError` and the whole transaction is rolled back, so nothing
        is committed (design choices entry 82). A refusal of the book raises ValueError with its sentence; nothing changed."""
        symbol = check_symbol(symbol)
        if side not in ("buy", "sell"):
            raise ValueError("side must be buy or sell")
        if isinstance(shares, bool) or not isinstance(shares, int) or shares < 1:
            raise ValueError("shares must be a whole number of at least 1")
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
            others_before = {name: row["shares"] for name, row in holdings.items() if name != symbol}
            cash = account["cash_balance"]
            held = position["shares"] if position else 0
            if side == "buy":
                cost = cents(price * shares)
                if cost <= 0:
                    raise ValueError("the trade must move at least one cent")
                if cash < cost:
                    raise ValueError(f"cash ${cash:,.2f} does not cover ${cost:,.2f}")
                _buy_position(cur, account_id, symbol, shares, price, traded_on, held_name, position)
                gain = Decimal("0.00")
                self._ledger(cur, account_id, "buy", -cost, cents(cash - cost), traded_on, f"buy {shares} {symbol}")
                want_cash, want_held = cents(cash - cost), held + shares
            else:
                if position is None or shares > held:
                    raise ValueError(f"cannot sell {shares} of {symbol}: the portfolio holds {held}")
                proceeds = cents(price * shares)
                if proceeds <= 0:
                    raise ValueError("the trade must move at least one cent")
                gain = proceeds - cents(position["cost_basis"] * shares)
                _sell_position(cur, account_id, symbol, shares, position)
                self._ledger(cur, account_id, "sell", proceeds, cents(cash + proceeds), traded_on, f"sell {shares} {symbol}")
                want_cash, want_held = cents(cash + proceeds), held - shares
            cur.execute("""INSERT INTO trading.trades (account_id, symbol, side, shares, price, realized_gain, traded_on, record_id, source)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING trade_id""",
                        (account_id, symbol, side, shares, price, gain, traded_on, None if record_id is None else record_id.strip(), source))
            row = cur.fetchone()["trade_id"]
            # the check, inside the transaction and before it commits: read the account again and compare it with what the trade must have done
            now_cash = self._cash(cur, account_id)
            cur.execute("SELECT symbol, shares FROM trading.positions WHERE account_id = %s", (account_id,))
            now_all = {r["symbol"]: r["shares"] for r in cur.fetchall()}
            now_held = now_all.get(symbol, 0)
            others_after = {name: n for name, n in now_all.items() if name != symbol}
            cur.execute("SELECT 1 FROM trading.trades WHERE trade_id = %s AND account_id = %s AND symbol = %s AND side = %s AND shares = %s", (row, account_id, symbol, side, shares))
            if now_cash != want_cash or now_held != want_held or others_after != others_before or cur.fetchone() is None:
                raise BookingCheckError(f"the account did not show the {side} of {shares} {symbol} when it was read again before the commit (cash ${now_cash:,.2f}, expected ${want_cash:,.2f}; "
                                        f"{now_held} shares, expected {want_held}{'' if others_after == others_before else '; another holding changed'}): the booking was rolled back and nothing was committed")
            return Booked(trade_row=row, gain=gain, symbol=symbol, side=side, shares=shares, price=price, cash_before=cash, cash_after=now_cash, held_before=held, held_after=now_held)
        return self._run(account_id, fn)

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
    shares: int
    price: Decimal
    cash_before: Decimal
    cash_after: Decimal
    held_before: int
    held_after: int


def _require_active(row: dict) -> None:
    if row["status"] != "active":
        raise ValueError(f"the account is {row['status']}: it cannot take funds or a trade")


def _buy_position(cur, account_id: int, symbol: str, shares: int, price: Decimal, traded_on: date, name: str | None, position: dict | None) -> None:
    if position is None:
        cur.execute("""INSERT INTO trading.positions (account_id, symbol, name, shares, cost_basis, acquired, recent_shares)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)""", (account_id, symbol, name, shares, price, traded_on, shares))
        return
    total = position["shares"] + shares
    basis = price_scale((position["shares"] * position["cost_basis"] + shares * price) / Decimal(total))
    if position["acquired"] is not None and position["acquired"] >= traded_on:
        locked = None if position["recent_shares"] is None else position["recent_shares"] + shares
    else:
        locked = shares
    cur.execute("""UPDATE trading.positions
                   SET name = COALESCE(%s, name), shares = %s, cost_basis = %s, acquired = %s, recent_shares = %s
                   WHERE account_id = %s AND symbol = %s""", (name, total, basis, traded_on, locked, account_id, symbol))


def _sell_position(cur, account_id: int, symbol: str, shares: int, position: dict) -> None:
    left = position["shares"] - shares
    if left == 0:
        cur.execute("DELETE FROM trading.positions WHERE account_id = %s AND symbol = %s", (account_id, symbol))
        return
    recent = position["recent_shares"]
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
    shares = decision.get("shares")
    if isinstance(shares, bool) or not isinstance(shares, int) or shares < 0:
        raise ValueError("a decision needs a share count")
    mode = decision.get("mode")
    if mode not in ("ask", "auto"):
        raise ValueError("a decision needs mode ask or auto")
    reason = decision.get("reason")
    if not isinstance(reason, str):
        raise ValueError("a decision needs a reason")
    return (account_id, record_id.strip(), check_symbol(decision.get("symbol") or ""), action, shares, mode, will_trade, reason, decided_on)


def _user(row: dict) -> UserAccount:
    horizon = row["investment_horizon_years"]
    return UserAccount(
        account_id=row["account_id"], display_name=row["display_name"], preferred_name=row["preferred_name"], email=row["email"],
        phone=row["phone"], timezone=row["timezone"], currency=row["currency"], status=row["status"], risk_tolerance=row["risk_tolerance"],
        trade_mode=row["trade_mode"], age=row["age"], investment_horizon_years=None if horizon is None else float(horizon), note=row["note"],
        seed_key=row["seed_key"], created_at=row["created_at"], updated_at=row["updated_at"])


def _db_message(error: Exception) -> str:
    diag = getattr(error, "diag", None)
    text = getattr(diag, "message_primary", None) or str(error)
    return text.splitlines()[0]
