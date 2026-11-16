"""Money, prices and the JSON encoder: the ONE home of the account's `decimal.Decimal` types and doors (design choices entry 112; the spec is design/decimal_money_spec.md).

    money(19.99)                  # Decimal('19.99')    a dollar amount to the cent
    price(158.32)                 # Decimal('158.3200') a price to four places, above zero (the account book's `numeric(14,4)`)
    rate(0.01)                    # Decimal('0.01')     a ratio, a percentage or a number of basis points, from the digits it is written with
    dumps({"cash": money("4907.92")})            # '{"cash": 4907.92}'   a Decimal is written as a JSON NUMBER, by its own digits, never as a string

THE RULES (Jacob, 2026-11-16: *"Need to always use decimal.Decimal type when dealing with money and share counts"*; his example, `Decimal("19.99")`, exact multiplication, then
`.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)`). In the account's code every dollar amount, every price a trade is sized or filled at and every share count (`shares.py`) is a `Decimal`. A `Decimal` is made from DIGITS,
never from a binary float: `Decimal("19.99")`, `Decimal(3)`, or the digits a number was written with (`Decimal(str(x))`); `Decimal(19.99)` is never written. A result is rounded once, on purpose, by name: money to the cent
(`cents`, `portfolio.trade_amount`), a price to four places (`price_scale`), a share count down to six places when it is bought with money (`shares.buys`). `Decimal * float` raises `TypeError`, and that is WANTED: a float
that was missed stops a test instead of putting a wrong cent in the account.

A NUMBER COMES IN THROUGH A DOOR, ONCE. `money`, `price`, `rate` (and `shares.quantity`) take an `int`, a `Decimal`, or a `float` by the digits it is written with, and refuse a bool and a string with the reason (a numeric
string such as "2" too: Jacob, 2026-11-15, *"Getting any numeric string like "2" is an error also"*), and anything that is not finite. A `float` that arrives from OUTSIDE (a JSON file or reply that a library read, a quote from
an API client, a stored bar, the Advisor's plan) is turned into a `Decimal` here and is never a float again. `money` does not round: a number with more than two decimals is refused (0.1 + 0.2 is `0.30000000000000004`), and
a caller that has a longer number rounds it with `cents` first, on purpose. `price` is the one rounding of a price: a quote with more than four decimals is rounded half up to four, because the account book holds four.

`Money`, `PositiveMoney`, `NonNegativeMoney`, `Price` and `Rate` are the pydantic types of a field that holds one: the door runs BEFORE pydantic's own conversion, so a bool and a string are refused (pydantic alone would turn
"2" into Decimal("2")), and in JSON mode they are written as NUMBERS (a whole value as an integer: `10000`, any other by its own digits: `63.163213`, `4907.92`).

THE ONE JSON ENCODER (D5: a money amount, a price and a share count are NUMBERS in JSON, as they always were, never strings; every capture still loads, and the page and the models still get numbers). `json_number` is the
only function that turns a `Decimal` into a number: a whole value is an `int`, any other is `float(d)`, and it first checks that `Decimal(repr(float(d))) == d` and raises when it is not (more digits than a float holds),
so a digit is never dropped in silence. `jsonable` does it through a whole structure, `json_default` is the `default=` of `json.dumps`, `dumps` is `json.dumps` with it, and `Jsonable` is the pydantic marker for a field that
holds plain dicts with Decimals in them (the work order's decisions and trades). Reading goes the other way through the typed doors: the float that `json` gives is turned into a `Decimal` by the field that holds it. A whole
work order is NEVER read with `parse_float=Decimal`: the Advisor's and the Critic's numbers in it are measurements and stay floats.
"""
import json
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Context, Decimal, InvalidOperation
from typing import Annotated, Any

from pydantic import BeforeValidator, PlainSerializer

from prometheus.common.user_account import cents, price_scale  # noqa: F401  (re-exported: the one family of rounding functions)

CENT = Decimal("0.01")
PRICE_STEP = Decimal("0.0001")
MONEY_PLACES = 2
PRICE_PLACES = 4
EXACT = Context(prec=60)                       # no rounding of a sum or a product by the context: an amount has at most 18 digits


def _number(x, what: str) -> Decimal:
    """`x` as the Decimal it is written as: an int, a Decimal, or a float by its digits (`Decimal(str(x))`, never the binary expansion). ValueError for a bool, a string, anything that is not a number, and a number that is not finite."""
    if isinstance(x, bool) or not isinstance(x, (int, float, Decimal)):
        raise ValueError(f"{what} is a number, not {x!r}")
    d = Decimal(x) if isinstance(x, int) else Decimal(str(x)) if isinstance(x, float) else x
    if not d.is_finite():
        raise ValueError(f"{what} is a finite number, not {x!r}")
    return d


def money(x) -> Decimal:
    """THE door of a dollar amount: `x` (an int, a Decimal or a float by its written digits; never a bool or a string) as a Decimal to the cent. More than two decimals is an error and is never rounded here: the caller that
    has a longer number rounds it with `cents` first, on purpose. May be negative (a gain, a difference); `positive_money` and `non_negative_money` say more."""
    d = _number(x, "an amount of money")
    try:
        scaled = d.quantize(CENT, context=EXACT)
    except InvalidOperation:
        raise ValueError(f"{x!r} is too large to be an amount of money") from None
    if d != scaled:
        raise ValueError(f"an amount of money has at most {MONEY_PLACES} decimals: {x!r} has more (round it with `cents`, on purpose)")
    return scaled


def positive_money(x) -> Decimal:
    d = money(x)
    if d <= 0:
        raise ValueError("an amount of money above zero is needed")
    return d


def non_negative_money(x) -> Decimal:
    d = money(x)
    if d < 0:
        raise ValueError("an amount of money of zero or more is needed")
    return d


def price(x) -> Decimal:
    """THE door of a price: `x` (an int, a Decimal or a float by its written digits) as a Decimal above zero at four places, half up. A price from OUTSIDE with more than four decimals (a quote, a stored bar, the Advisor's plan)
    is rounded to four places here, because the account book holds four (`numeric(14,4)`): this is the one rounding of a price."""
    d = _number(x, "a price")
    if d <= 0:
        raise ValueError("a price above zero is needed")
    try:
        return d.quantize(PRICE_STEP, rounding=ROUND_HALF_UP, context=EXACT)
    except InvalidOperation:
        raise ValueError(f"{x!r} is too large to be a price") from None


def rate(x) -> Decimal:
    """A ratio, a percentage or a number of basis points (the Trader's `max_chase_pct`, `limit_band_pct` and `slippage_bps`, a percentage the user typed) as the Decimal of the digits it is written with. Zero or more; never a bool or a string."""
    d = _number(x, "a rate")
    if d < 0:
        raise ValueError("a rate of zero or more is needed")
    return d


def canonical(d: Decimal) -> Decimal:
    """The one form of a plain number (a count, an amount the user typed): a whole value with no decimals (Decimal('20'), also for 20.0 and 2E+1), any other without trailing zeros (Decimal('2.5')), so that it prints and is written as it was said."""
    return Decimal(int(d)) if d == d.to_integral_value() else d.normalize(EXACT)


def digits(x) -> Decimal:
    """Any finite number (an int, a Decimal or a float by its written digits) as a Decimal, for a place that is none of the kinds above (a number of the user's words, a model's reply that was checked against them). Never a bool or a string."""
    return _number(x, "a number")


def exact(x, what: str = "a number") -> Decimal:
    """`x` for the arithmetic INSIDE the account's code (a function that takes a price or an amount it was given, after the doors): a Decimal or an int, as it is. A float here is a mistake of the program (it should have come in
    through a door: `money`, `price`, `rate`, `shares.quantity`) and raises TypeError, which is what stops a test instead of putting a wrong cent in the account; a bool or anything else that is not a number is a ValueError."""
    if isinstance(x, float):
        raise TypeError(f"{what} must be a Decimal, not a float ({x!r}): a float comes in through a door (money, price, rate, shares.quantity), once")
    if isinstance(x, bool) or not isinstance(x, (int, Decimal)):
        raise ValueError(f"{what} is a number, not {x!r}")
    d = Decimal(x) if isinstance(x, int) else x
    if not d.is_finite():
        raise ValueError(f"{what} is a finite number, not {x!r}")
    return d


# ---------------------------------------------------------------- the one JSON encoder

def json_number(d: Decimal) -> int | float:
    """The JSON NUMBER of a Decimal: an `int` when it is whole (81, 10000), else the `float` with the same digits (63.163213, 4907.92). Raises ValueError, and does not write anything, when a float cannot hold its digits
    (`Decimal(repr(float(d))) != d`), so a digit is never dropped in silence."""
    if not isinstance(d, Decimal) or not d.is_finite():
        raise ValueError(f"only a finite Decimal is written as a number, not {d!r}")
    if d == d.to_integral_value():
        return int(d)
    f = float(d)
    if Decimal(repr(f)) != d:
        raise ValueError(f"{d} has more digits than a JSON number holds and would be changed to {f!r}")
    return f


def jsonable(value):
    """`value` with every Decimal in it turned into a JSON number by `json_number`, through dicts, lists and tuples; anything else as it is."""
    if isinstance(value, Decimal):
        return json_number(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def json_default(o):
    """The `default=` of `json.dumps` for the account's values: a Decimal is a number (`json_number`), a date its ISO text, anything else its text (what `default=str` did for the other kinds). Never turns a Decimal into text."""
    if isinstance(o, Decimal):
        return json_number(o)
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    return str(o)


def dumps(obj, **kwargs) -> str:
    """`json.dumps` with the one encoder: a Decimal anywhere in `obj` is written as a number."""
    return json.dumps(obj, default=json_default, **kwargs)


Number = int | float
_NUMBER = PlainSerializer(json_number, return_type=Number, when_used="json")
Jsonable = PlainSerializer(jsonable, return_type=Any, when_used="json")      # for a field of plain dicts and lists that may hold Decimals (the work order's decisions and trades)

Money = Annotated[Decimal, BeforeValidator(money), _NUMBER]
PositiveMoney = Annotated[Decimal, BeforeValidator(positive_money), _NUMBER]
NonNegativeMoney = Annotated[Decimal, BeforeValidator(non_negative_money), _NUMBER]
Price = Annotated[Decimal, BeforeValidator(price), _NUMBER]
Rate = Annotated[Decimal, BeforeValidator(rate), _NUMBER]
Digits = Annotated[Decimal, BeforeValidator(digits), _NUMBER]
