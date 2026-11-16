"""A number of shares: the one place that reads, adds, compares or prints a share quantity (design choices entry 110; the spec is design/fractional_shares_spec.md; since design choices entry 112 a quantity is a `decimal.Decimal`,
design/decimal_money_spec.md). Plain Python: no model, no network.

    quantity(81)            # Decimal('81')            a whole number: a Decimal with no fraction (it was a Python int before entry 112)
    quantity(81.004455)     # Decimal('81.004455')     a float is taken by the digits it is written with and has at most PLACES (6) decimals
    quantity(0.1234567)     # ValueError: more than 6 decimals is refused, never rounded
    plus(81, 0.004455)      # Decimal('81.004455')     quantities are added and taken apart exactly, in Decimal
    minus(81.004455, 81)    # Decimal('0.004455')
    buys(Decimal("10000.00"), Decimal("123.45"))   # the most shares (at STEP) that $10,000.00 pays for at $123.45, the way the account book rounds a trade's cost
    text(Decimal("1234.5")) # '1,234.5'                what every sentence, report and page prints

A share quantity has at most PLACES decimals (F1): 0.000001 share is the smallest step, and the Postgres columns that hold one are `numeric(18,6)` (sql/prometheus_accounts.sql; a test fails when the two differ). A quantity is a
`Decimal` everywhere (entry 112, Jacob: *"Need to always use decimal.Decimal type when dealing with money and share counts"*), in canonical form: a whole quantity has no decimals (`Decimal('81')`, never `81.000000`), a fraction
no trailing zero (`Decimal('81.00445')`), so that it prints and is written to JSON as it always was: `81`, `81.004455`. Two things produce a fraction (F3): the Account Manager's even split of the user's money and a direct order in
which the user typed a decimal number of shares; nothing here makes one by itself. Money is not touched: `portfolio.trade_amount` is still the one rounding of a price times shares to the cent.

`quantity` is the door: it takes an int, a float (by the digits it is written with) or a Decimal, and refuses a bool and a string (a numeric string such as "2" too). `Shares` is the type of a field that holds a quantity (zero or
more) and `PositiveShares` one that holds a quantity above zero. Both run `quantity` BEFORE pydantic's own conversion, so a bool and a string are refused, and in JSON mode they are written as NUMBERS (a whole quantity as an integer).
"""
import math
from decimal import ROUND_DOWN, ROUND_HALF_UP, Context, Decimal, InvalidOperation
from fractions import Fraction
from typing import Annotated

from pydantic import AfterValidator, BeforeValidator, PlainSerializer

from prometheus.common.money import canonical, json_number

PLACES = 6
STEP = Decimal("0.000001")                       # the smallest quantity there is; a test reads sql/prometheus_accounts.sql and fails if its columns' scale is not PLACES
_CENT = Decimal("0.01")
_EXACT = Context(prec=60)                        # no rounding of a sum or a product of quantities: a quantity has at most 18 digits


def _decimal(x) -> Decimal:
    """`x` as the Decimal it is written as (`Decimal(str(x))`, never the binary float's long expansion), or ValueError for what is not a number: a bool, a string, anything that is not an int, a float or a Decimal."""
    if isinstance(x, bool) or not isinstance(x, (int, float, Decimal)):
        raise ValueError(f"a number of shares is a number, not {x!r}")
    d = Decimal(x) if isinstance(x, int) else Decimal(str(x)) if isinstance(x, float) else x
    if not d.is_finite():
        raise ValueError(f"a number of shares is finite, not {x!r}")
    return d


_canonical = canonical      # the one form of a quantity: a whole value with no decimals (Decimal('81'), also for 81.000000 and 0E-6), any other without trailing zeros (Decimal('81.00445'))


def quantity(x) -> Decimal:
    """THE door of a share quantity. `x` is an int, a float or a Decimal (never a bool or a string) that is finite, zero or more, and has at most PLACES decimals (counted from the digits it is written with: 0.1234567 is refused,
    not rounded). Returns the Decimal in its canonical form (a whole number has no decimals: 81.0 is `Decimal('81')`). Raises ValueError, with the reason, otherwise."""
    d = _decimal(x)
    if d < 0:
        raise ValueError(f"a number of shares is zero or more, not {x!r}")
    try:
        scaled = d.quantize(STEP, context=_EXACT)
    except InvalidOperation:
        raise ValueError(f"{x!r} is too large to be a number of shares") from None
    if d != scaled:
        raise ValueError(f"a number of shares has at most {PLACES} decimals: {x!r} has more")
    return _canonical(d)


def _positive(x):
    if x <= 0:
        raise ValueError("a number of shares above zero is needed")
    return x


_NUMBER = PlainSerializer(json_number, return_type=int | float, when_used="json")
Shares = Annotated[Decimal, BeforeValidator(quantity), _NUMBER]
PositiveShares = Annotated[Decimal, BeforeValidator(quantity), AfterValidator(_positive), _NUMBER]


def as_decimal(x) -> Decimal:
    """`x` (a quantity) at scale PLACES, for the account book and for exact arithmetic: Decimal('81.004455'), Decimal('81.000000')."""
    return quantity(x).quantize(STEP)


def plus(a, b) -> Decimal:
    """`a` and `b` added exactly, as a quantity: 81 plus 0.004455 is 81.004455, and 0.1 plus 0.2 is 0.3."""
    return quantity(_EXACT.add(quantity(a), quantity(b)))


def minus(a, b) -> Decimal:
    """`a` less `b`, exactly, as a quantity: 81.004455 less 81 is exactly 0.004455, and a holding sold in full is exactly 0. Raises ValueError when `b` is more than `a`."""
    left = _EXACT.subtract(quantity(a), quantity(b))
    if left < 0:
        raise ValueError(f"cannot take {text(b)} from {text(a)}")
    return quantity(left)


def difference(a, b) -> Decimal:
    """`a` less `b`, exactly, SIGNED: the change of a holding from `b` to `a` (negative when it fell). Not a quantity (it may be below zero), but in the same canonical form."""
    return _canonical(_EXACT.subtract(quantity(a), quantity(b)))


def same(a, b) -> bool:
    """Whether two quantities are equal (a quantity is a Decimal, so `==` says it; this also takes the int and the float that a JSON file gives, by their digits)."""
    return quantity(a) == quantity(b)


def whole(x) -> bool:
    """Whether the quantity is a whole number of shares."""
    d = quantity(x)
    return d == d.to_integral_value()


def floor(x) -> Decimal:
    """The whole shares in the quantity."""
    return Decimal(math.floor(quantity(x)))


def ceil(x) -> Decimal:
    """The quantity rounded UP to the next whole share (a whole quantity is itself)."""
    return Decimal(math.ceil(quantity(x)))


def buys(dollars, price) -> Decimal:
    """The shares (at STEP) that `dollars` buys at `price`: the most whose price times shares is not above `dollars`, exactly (`dollars` divided by `price` and rounded DOWN to the step, in exact arithmetic), so that its cost,
    `portfolio.trade_amount(shares, price)` (rounded half up to the cent, the way the account book does it), is `dollars` to the cent whenever one step costs less than half a cent (a share price under $5,000). Only when
    the step costs more (a price from $5,000 to $10,000) the rounding can leave the cost a cent short, and then one step MORE is taken if that costs no more than `dollars`; so for a price under $10,000 the cost is `dollars`
    to the cent, and it is never above it. 0 when nothing fits (a trade has to cost at least one cent). Closed form, never a loop of steps. `dollars` is taken to the cent, below it; `dollars` and `price` are numbers above
    zero, else 0. A whole result has no decimals: $1,000 at $1,000 buys exactly 1."""
    try:
        cap, rate = _decimal(dollars), _decimal(price)
    except ValueError:
        return Decimal(0)
    if rate <= 0 or cap <= 0:
        return Decimal(0)
    cap = cap.quantize(_CENT, rounding=ROUND_DOWN, context=_EXACT)
    if cap < _CENT:
        return Decimal(0)
    steps = math.floor(Fraction(cap) / (Fraction(rate) * Fraction(STEP)))                # the exact quotient, in whole steps, rounded down
    cost = lambda n: _cost(n, rate)
    if steps > 0 and cost(steps) < cap and cost(steps + 1) <= cap:                       # the cent rounding left it short: one more step is still within the dollars
        steps += 1
    while steps > 0 and cost(steps) > cap:                                               # (a rounding the quotient cannot see; never more than one step)
        steps -= 1
    return quantity(Decimal(steps) * STEP) if steps > 0 and cost(steps) >= _CENT else Decimal(0)


def _cost(steps: int, rate: Decimal) -> Decimal:
    """What `steps` of STEP cost at `rate`, in cents, half up (`portfolio.trade_amount` on the same numbers, kept here so that this module imports nothing of the portfolio)."""
    return (rate * Decimal(steps) * STEP).quantize(_CENT, rounding=ROUND_HALF_UP, context=_EXACT)


def text(x) -> str:
    """A quantity as every sentence, report and page prints it: thousands separators and the decimals as they are, no trailing zero: '81', '1,234', '81.004455', '0.5'."""
    q = quantity(x)
    if q == q.to_integral_value():
        return f"{int(q):,}"
    whole_part, _, decimals = format(q, "f").partition(".")
    return f"{int(whole_part):,}.{decimals}"


def shown(x) -> str:
    """`text` for a quantity, and the value as it is for anything else: for a message about a number that may be what is wrong with the request (a bool, a string, seven decimals)."""
    try:
        return text(x)
    except ValueError:
        return str(x)


def signed(x) -> str:
    """`text` with a sign, for a change in a holding (which may be below zero): '+0.5', '-81', '+0'."""
    d = _decimal(x)
    return ("-" if d < 0 else "+") + text(abs(d))


def noun(x) -> str:
    """'share' for exactly one, else 'shares' (0.5 shares, 1 share, 2 shares)."""
    return "share" if quantity(x) == 1 else "shares"
