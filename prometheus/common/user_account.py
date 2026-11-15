"""The account holder: the name, the account number, and the fields a later screen will show (design choices entry 72).

    holder.account_number          # the database id; it is never changed
    holder.display_name            # the same text as the portfolio's user and the profile's user
    directory.set_name("Avery Chen")
    directory.set_email("avery.chen@example.com")

The risk levels a person can set are the three already on the profile: conservative, moderate, aggressive
(`user_profile.TOLERANCES`). The Critic's low, medium, high and very_high describe a recommendation, not a setting.
Nothing here calls an API or a model. The rows live in Postgres; this module only checks the values.
"""
import re
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field

from prometheus.common.user_profile import TOLERANCES, Tolerance

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE = re.compile(r"^[0-9+(). -]{7,40}$")
STATUSES = ("active", "suspended", "closed")
TRADE_MODES = ("ask", "auto")


class UserAccount(BaseModel):
    """One saved account. `account_id` is assigned by the database."""
    model_config = ConfigDict(extra="forbid")

    account_id: int = Field(ge=1)
    display_name: str
    preferred_name: str | None = None
    email: str | None = None
    phone: str | None = None
    timezone: str
    currency: str
    status: str
    risk_tolerance: Tolerance
    trade_mode: str
    age: int | None = None
    investment_horizon_years: float | None = None
    note: str | None = None
    seed_key: str | None = None
    created_at: datetime
    updated_at: datetime

    @property
    def account_number(self) -> int:
        """The account id. A screen can show this; nothing writes it."""
        return self.account_id


class UserDirectory:
    """Read and change the holder's name and the fields the screen will show. The account number does not change."""

    def __init__(self, store, account_id: int, on_change=None):
        if isinstance(account_id, bool) or not isinstance(account_id, int) or account_id < 1:
            raise ValueError("account_id must be a positive whole number")
        self._store = store
        self._account_id = account_id
        self._on_change = on_change

    @property
    def account_id(self) -> int:
        return self._account_id

    @property
    def account_number(self) -> int:
        return self._account_id

    def get(self) -> UserAccount:
        return self._store.get_user(self._account_id)

    def set_name(self, display_name: str) -> UserAccount:
        return self._save(display_name=display_name)

    def set_preferred_name(self, preferred_name: str | None) -> UserAccount:
        return self._save(preferred_name=preferred_name)

    def set_email(self, email: str | None) -> UserAccount:
        return self._save(email=email)

    def set_phone(self, phone: str | None) -> UserAccount:
        return self._save(phone=phone)

    def set_timezone(self, timezone: str) -> UserAccount:
        return self._save(timezone=timezone)

    def set_note(self, note: str | None) -> UserAccount:
        return self._save(note=note)

    def set_status(self, status: str) -> UserAccount:
        return self._save(status=status)

    def set_trade_mode(self, trade_mode: str) -> UserAccount:
        return self._save(trade_mode=trade_mode)

    def set_age(self, age: int | None) -> UserAccount:
        return self._save(age=age)

    def set_horizon(self, years) -> UserAccount:
        return self._save(investment_horizon_years=years)

    def _save(self, **changes) -> UserAccount:
        saved = self._store.update_user(self._account_id, **changes)
        if self._on_change is not None:
            self._on_change()
        return saved


def clean_display_name(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("the name must be text")
    cleaned = " ".join(value.split())
    if not cleaned or len(cleaned) > 200 or any(ord(ch) < 32 for ch in cleaned):
        raise ValueError("the name must be from 1 to 200 characters")
    return cleaned


def clean_optional_name(value: str | None, *, limit: int = 80) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("the name must be text")
    cleaned = " ".join(value.split())
    if not cleaned:
        return None
    if len(cleaned) > limit or any(ord(ch) < 32 for ch in cleaned):
        raise ValueError(f"the name must be at most {limit} characters")
    return cleaned


def clean_email(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("email must be text")
    cleaned = value.strip()
    if not cleaned:
        return None
    if _EMAIL.fullmatch(cleaned) is None or len(cleaned) > 200:
        raise ValueError("email must look like a name at a domain")
    return cleaned


def clean_phone(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("phone must be text")
    cleaned = " ".join(value.split())
    if not cleaned:
        return None
    if _PHONE.fullmatch(cleaned) is None:
        raise ValueError("phone may contain digits, spaces, and + ( ) . -")
    return cleaned


def clean_timezone(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("timezone must be text, such as America/New_York")
    try:
        ZoneInfo(value.strip())
    except ZoneInfoNotFoundError:
        raise ValueError(f"unknown timezone {value.strip()!r}") from None
    return value.strip()


def clean_note(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("the note must be text")
    cleaned = value.strip()
    if not cleaned:
        return None
    if len(cleaned) > 2000:
        raise ValueError("the note must be at most 2000 characters")
    return cleaned


def clean_status(value: str) -> str:
    if not isinstance(value, str) or value.strip().lower() not in STATUSES:
        raise ValueError(f"status must be one of {', '.join(STATUSES)}")
    return value.strip().lower()


def clean_trade_mode(value: str) -> str:
    if not isinstance(value, str) or value.strip().lower() not in TRADE_MODES:
        raise ValueError(f"trade mode must be one of {', '.join(TRADE_MODES)}")
    return value.strip().lower()


def clean_risk_level(level: str) -> str:
    """One of conservative, moderate, aggressive. Anything else, including a Critic level such as low, is refused."""
    if not isinstance(level, str) or level.strip().lower() not in TOLERANCES:
        raise ValueError(f"risk level must be one of {', '.join(TOLERANCES)}")
    return level.strip().lower()


def clean_age(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 18 <= value <= 120:
        raise ValueError("age must be a whole number from 18 to 120, or empty")
    return value


def clean_horizon(value) -> Decimal | None:
    if value is None:
        return None
    number = _plain_decimal(value, "the horizon")
    scaled = number.quantize(Decimal("0.01"))
    if scaled <= 0:
        raise ValueError("the horizon must be greater than zero")
    return scaled


def parse_money(amount) -> Decimal:
    """A positive number of dollars with at most two decimal places. A float is refused."""
    value = _plain_decimal(amount, "amount")
    cents = value.quantize(Decimal("0.01"))
    if cents != value:
        raise ValueError("amount can have at most two decimal places")
    if cents <= 0:
        raise ValueError("amount must be greater than zero")
    return cents


def parse_price(price) -> Decimal:
    """A share price with at most four decimal places."""
    value = _plain_decimal(price, "price")
    scaled = value.quantize(Decimal("0.0001"))
    if scaled != value or scaled <= 0:
        raise ValueError("price must be greater than zero and have at most four decimal places")
    return scaled


def cents(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def price_scale(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


def date_window(start, end) -> tuple[date, date]:
    """Both ends are included. A datetime is refused: pass a date."""
    start, end = _one_date(start, "start"), _one_date(end, "end")
    if start > end:
        raise ValueError("start must be on or before end")
    return start, end


def _one_date(value, label: str) -> date:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise ValueError(f"{label} must be a date")
    return value


def _plain_decimal(value, label: str) -> Decimal:
    if isinstance(value, bool) or isinstance(value, float) or not isinstance(value, (Decimal, int, str)):
        raise ValueError(f"{label} must be a decimal number, such as Decimal('100.00')")
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text or any(ch in text for ch in "eE"):
            raise ValueError(f"{label} must be a decimal number, such as Decimal('100.00')")
        try:
            number = Decimal(text)
        except InvalidOperation:
            raise ValueError(f"{label} must be a decimal number, such as Decimal('100.00')") from None
    else:
        number = Decimal(int(value)) if not isinstance(value, Decimal) else value
    if not number.is_finite():
        raise ValueError(f"{label} must be a finite number")
    return number
