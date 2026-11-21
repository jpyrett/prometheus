"""A user's portfolio: what the Account Manager reads when it has to raise funds (design choices entry 42).

    portfolio = load_portfolio("configs/portfolios/jacob_pyrett.json")
    portfolio.position("TSLA").shares                       # 123
    portfolio.sellable_shares("TSLA", today=date.today())   # shares that may be sold today: not bought today (the design's T+1 holding rule)

The JSON files are still what the desk loads by default. The account book (design choices entry 72) loads this same class from Postgres. Nothing here calls an API or an LLM.

A NUMBER OF SHARES MAY HAVE A FRACTION (design choices entry 110): `Position.shares` and `recent_shares` are `shares.PositiveShares` (a Decimal with at most six decimals), quantities are added and taken apart
with `shares.plus` and `shares.minus` (exact), a position is removed when exactly 0 is left, and `trade_amount` is the one rounding of money, to the cent, whatever the quantity. `"shares": 81.004455` in a JSON file is a valid holding.

MONEY AND PRICES ARE `decimal.Decimal` (design choices entry 112; design/decimal_money_spec.md): `Portfolio.cash_avail` is a `Cash` (a Decimal to the cent, zero or more), `Position.cost_basis` a `money.Price` (four places, above
zero), and every sum, difference and product is Decimal arithmetic that is rounded once, on purpose: `trade_amount` (a price times shares, to the cent, half up), `_average_cost` (`price_scale`). A float in the arithmetic raises
TypeError (`money.exact`): it comes in through a door, once. The JSON file's numbers are floats once `json` has read them and the fields turn them into Decimals by the digits they are written with.

    {"schema_version": 1, "user": "Jacob Pyrett", "currency": "USD", "cash_avail": "51.23",
     "positions": [{"symbol": "MSFT", "name": "Microsoft Corp", "shares": 50, "cost_basis": 150.0, "acquired": "2024-03-15"}, ...]}

`cash_avail` is the FREE SETTLED CASH: the money that can buy today. RaiseFunds spends it first, and only sells what is still missing. It may be a number or a string of a number
("51.23"), as in the sample file (the ONE string that is read on purpose: a portfolio FILE may write its cash as text); it must be at least 0, finite and to the cent. `cost_basis` is the average price paid per share. `acquired` is the date of the latest purchase of the position (left out: all of it is sellable).
"""
import json
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, Field, PlainSerializer, field_validator, model_validator

from prometheus.common import shares as sh
from prometheus.common import trading_clock
from prometheus.common.money import EXACT, Price, cents, exact, json_number, non_negative_money, price_scale
from prometheus.common.shares import PositiveShares, Shares
from prometheus.rag.loaders import check_symbol


def _cash_text(value):
    """A portfolio FILE may write its cash as text ("51.23", as in the sample file): the one string that is read on purpose, parsed from its digits here, at the field. Any other value goes to the door of money as it is (a bool is refused)."""
    if isinstance(value, str):
        try:
            value = Decimal(value.strip().replace(",", ""))
        except InvalidOperation:
            raise ValueError(f"cash_avail must be a number of dollars, not {value!r}") from None
    return non_negative_money(value)


Cash = Annotated[Decimal, BeforeValidator(_cash_text), PlainSerializer(json_number, return_type=int | float, when_used="json")]


class Position(BaseModel):
    symbol: str
    name: str | None = None
    shares: PositiveShares                       # a Decimal (a whole number has no decimals, a fraction has at most 6: prometheus/common/shares.py)
    cost_basis: Price                            # average price paid per share, four places, above zero
    acquired: date | None = None                 # when the latest shares were bought; None means long ago
    recent_shares: PositiveShares | None = None  # how many of the shares were bought on `acquired` (the T+1 lock applies to these only); None: all of the position

    @field_validator("symbol")
    @classmethod
    def _symbol(cls, value: str) -> str:
        return check_symbol(value.strip().lstrip("$"))


class Portfolio(BaseModel):
    schema_version: int = 1
    user: str
    currency: str = "USD"
    cash_avail: Cash = Decimal("0.00")                                  # free settled cash, in `currency`, to the cent: what can buy today. Sale proceeds that have not settled (T+1) are not in it, and neither is the cash a waiting order has SET ASIDE (below). A file may write it as text ("51.23")
    positions: list[Position] = Field(default_factory=list)
    note: str | None = None
    reserved_cash: Cash = Field(default=Decimal("0.00"), exclude=True)  # WAITING ORDERS (design choices entry 140): the cash the account's open orders hold, which `cash_avail` has already had taken off (the account book's `list_portfolio` reads the one door, `AccountStore.available_cash`). Not part of any dump: only the account book knows it
    reserved_shares: dict[str, Shares] = Field(default_factory=dict, exclude=True)  # {symbol: shares} the open orders to SELL hold: `positions` keeps the whole holding, and `sellable_shares` takes these off

    @model_validator(mode="after")
    def _one_line_per_symbol(self):
        symbols = [p.symbol for p in self.positions]
        if len(symbols) != len(set(symbols)):
            raise ValueError("a symbol appears twice in the portfolio")
        return self

    @property
    def symbols(self) -> list[str]:
        return [p.symbol for p in self.positions]

    def position(self, symbol: str) -> Position | None:
        return next((p for p in self.positions if p.symbol == symbol), None)

    def sellable_shares(self, symbol: str, today: date | None = None) -> Decimal:
        """The shares of `symbol` that may be sold today. The design's T+1 holding rule: shares bought today are not sellable until tomorrow. If the position says how many of its
        shares were bought on `acquired` (`recent_shares`), only those are locked and the older ones are sellable; without it the whole position is locked. A position whose purchase
        date is not given is all sellable."""
        position = self.position(symbol)
        if position is None:
            return Decimal(0)
        today = today or trading_clock.market_day()
        if position.acquired is None or position.acquired < today:
            free = position.shares
        else:
            free = Decimal(0) if position.recent_shares is None else sh.minus(position.shares, min(position.recent_shares, position.shares))
        return min(free, self.free_shares(symbol))      # the shares a waiting sale has set aside may not be sold again (design choices entry 140)

    @property
    def cash_total(self) -> Decimal:
        """ALL the cash of the account: the available cash and what waiting orders have set aside. What the account holds, as the book's ledger has it; `cash_avail` is what may be spent."""
        return self.cash_avail + self.reserved_cash

    def free_shares(self, symbol: str) -> Decimal:
        """The shares of `symbol` held that no waiting sale has set aside: what a new sale may sell (0 when none is held)."""
        position = self.position(symbol)
        if position is None:
            return Decimal(0)
        reserved = self.reserved_shares.get(symbol, Decimal(0))
        return Decimal(0) if reserved >= position.shares else sh.minus(position.shares, reserved)

    def market_value(self, prices: dict[str, Decimal]) -> Decimal:
        return sum((trade_amount(p.shares, prices[p.symbol]) for p in self.positions), Decimal("0.00"))      # each holding's value is `trade_amount`: decimal, half up to the cent (design choices entries 111 and 112)


def trade_amount(shares, price) -> Decimal:
    """What `shares` at `price` cost, or bring in, in cents: the way the account book rounds it (the exact product, half up to the cent, on the price as it is written: 500.125 x 1 is 500.13, where Python's own `round`
    gives 500.12). `apply_fill`, the Trader's paper broker and a fill's `amount` all use it, so that the portfolio in memory, the fill and the booked cash never differ by a cent. `shares` is a quantity and may have a
    fraction (design choices entry 110); `price` is a Decimal (or an int): a float raises TypeError, it comes in through a door (design choices entry 112). The product is exact before the one rounding: 81.004455 shares at
    123.45 is worked out in full, then rounded."""
    return cents(EXACT.multiply(sh.quantity(shares), exact(price, "a price")))


def affordable_shares(cash, price, shares, commission=Decimal(0)) -> Decimal:
    """How many of `shares` the cash pays for at `price`, the way the account does the sum. For a WHOLE `shares`, the most whole shares `n` (0 to `shares`) with `cash - trade_amount(n, price) - commission >= 0`
    (an amount is rounded half up to the cent, `trade_amount`, and the account refuses a purchase the cash does not cover to the cent): 0 when not even one share is affordable. For a `shares` with a FRACTION (design choices entry 110:
    the even split, or a decimal the user typed in a direct order): `min(shares, shares.buys(cash - commission, price))`, the most of that quantity the cash pays for, at six decimals; so a whole order is cut to whole shares and a
    fractional order to a fractional quantity, and neither is ever cut by a loop of steps. This is `orders.cash_policy: "reduce"` of the Trader: buying fewer shares than approved is inside the permission, buying more never is. The
    Account Manager's `approve_trade` is the only caller (design choices entry 81: the money is the Account Manager's, the Trader never works it out); it lived in the Trader's `simulator.py` until then."""
    try:
        wanted = sh.quantity(shares)
    except ValueError:
        return Decimal(0)
    try:
        cash, price, commission = exact(cash, "the cash"), exact(price, "a price"), exact(commission, "a commission")      # a float is a TypeError (it comes in through a door)
    except ValueError:                                  # a cash or a price that is not a number (None, NaN, a bool): nothing can be bought with it, as it always was
        return Decimal(0)
    if price <= 0 or wanted <= 0 or cash <= 0:
        return Decimal(0)
    if not sh.whole(wanted):
        return min(wanted, sh.buys(cash - commission, price))
    n = min(wanted, max(cash - commission, Decimal(0)) // price + 1)       # the quotient is exact: start one above it and step down to the first that fits
    while n > 0 and cash - trade_amount(n, price) - commission < 0:
        n -= 1
    return n


def _sell_from(held: dict[str, Position], symbol: str, shares) -> None:
    """Take `shares` of `symbol` out of `held` (a dict of position copies, changed in place); a position sold out completely (exactly 0 left) is removed. The locked shares are sold last: a caller only sells what is
    sellable, so this only keeps `recent_shares` inside the position. Raises ValueError for a sell the portfolio cannot make (a symbol it does not hold, a quantity that is not above zero or is more than the shares held)."""
    position = held.get(symbol)
    try:
        wanted = sh.quantity(shares)
    except ValueError:
        wanted = Decimal(0)
    if position is None or wanted <= 0 or wanted > position.shares:
        raise ValueError(f"cannot sell {shares} of {symbol}: the portfolio holds {sh.text(position.shares) if position else 0}")
    position.shares = sh.minus(position.shares, wanted)
    if position.recent_shares is not None:
        position.recent_shares = min(position.recent_shares, position.shares) or None
    if position.shares == 0:
        del held[symbol]


def _buy_into(held: dict[str, Position], symbol: str, shares: Decimal, price: Decimal, today: date, name: str | None = None) -> None:
    """Add `shares` of `symbol` bought at `price` on `today` to `held` (a dict of position copies, changed in place): THE rules of the T+1 lock and of the cost basis, in one place (`after_purchase` and
    `apply_fill` both use it). A new position is all bought today, so all of it is locked. A symbol already held gets the weighted average cost basis, `acquired` becomes today, and only the new shares
    are locked (the older ones stay sellable), unless the position was already wholly locked, which it stays. `name` is kept when the position has one and none is given. Quantities are added with `shares.plus`."""
    existing = held.get(symbol)
    if existing is None:
        held[symbol] = Position(symbol=symbol, name=name, shares=shares, cost_basis=exact(price, "a price"), acquired=today)  # a new position: all of it was bought today, so all of it is locked
        return
    total = sh.plus(existing.shares, shares)
    if existing.acquired is not None and existing.acquired >= today:  # it was already bought into today
        locked = None if existing.recent_shares is None else sh.plus(existing.recent_shares, shares)  # None: the whole position is locked, and stays so
    else:
        locked = sh.quantity(shares)  # the older shares stay sellable; only the new ones are locked
    held[symbol] = existing.model_copy(update={"shares": total, "cost_basis": _average_cost(existing.shares, existing.cost_basis, shares, price, total),
                                               "acquired": today, "recent_shares": locked, "name": name or existing.name})


def _average_cost(held: Decimal, basis: Decimal, bought: Decimal, price: Decimal, total: Decimal) -> Decimal:
    """The weighted average cost of a position after a purchase, to 4 places, in decimal arithmetic (`price_scale`, half up: the way the account book works it, `accounts._buy_position`)."""
    paid = EXACT.add(EXACT.multiply(sh.quantity(held), exact(basis, "a cost basis")), EXACT.multiply(sh.quantity(bought), exact(price, "a price")))
    return price_scale(EXACT.divide(paid, sh.quantity(total)))


def _with_positions(portfolio: Portfolio, held: dict[str, Position], cash: Decimal) -> Portfolio:
    """A NEW portfolio with these positions (the symbols the portfolio already had keep their place, new ones come last) and this cash."""
    order = [s for s in portfolio.symbols if s in held] + [s for s in held if s not in portfolio.symbols]
    return portfolio.model_copy(update={"positions": [held[s] for s in order], "cash_avail": cents(exact(cash, "the cash"))})


def after_purchase(portfolio: Portfolio, target: str, shares: Decimal, price: Decimal, sells: list[tuple[str, Decimal]], cash_after: Decimal, today: date | None = None) -> Portfolio:
    """The portfolio once a plan has been carried out: the `sells` taken out (a position sold out completely is removed), `shares` of `target` bought at `price`, `cash_avail` set to `cash_after`
    (what the plan says is left). The shares bought are marked as bought `today` (`acquired`, `recent_shares`), so that the T+1 lock keeps them from being sold in the next plan of the same day,
    and the older shares of the same stock stay sellable. A target already held gets the weighted average cost basis. Returns a NEW portfolio: the one given is not changed. Raises
    ValueError for a sell the portfolio cannot make (a symbol it does not hold, more shares than held)."""
    today = today or trading_clock.market_day()
    held = {p.symbol: p.model_copy() for p in portfolio.positions}
    for symbol, n in sells:
        _sell_from(held, symbol, n)
    _buy_into(held, target, shares, price, today)
    return _with_positions(portfolio, held, cash_after)


def apply_fill(portfolio: Portfolio, symbol: str, side: str, shares: Decimal, price: Decimal, today: date, name: str | None = None) -> Portfolio:
    """The portfolio after ONE filled trade (Trader v1, design/trader_v1_final_spec.md section 13): the Account Manager's `notify_trades` and the Trader's paper broker both call it, so they cannot disagree.

    A buy costs `trade_amount(shares, price)` (the exact product, half up to the cent, as the account book rounds), which the cash must cover; the position follows the rules `after_purchase` has (the average cost basis, `acquired`, `recent_shares` for the T+1 lock: the same code).
    A sell needs the shares to be held; the proceeds, `trade_amount(shares, price)`, are added to the cash, the position keeps `recent_shares` inside itself, and a position sold out is removed. A trade must move
    at least one cent. `name` names a position that is new (or has no name). Returns a NEW portfolio; the one given is not changed. Raises ValueError, with the reason, for anything it cannot apply:
    a side that is not buy or sell, a quantity that is not a number above zero (a whole number, or one with up to six decimals: `shares.quantity`), a price that is not a finite number above zero, a date that is not a date, a symbol that is not a ticker, a purchase the cash does not cover, a sale of shares that are not held.
    It does not check the T+1 lock of a sale: the Trader's tickets are already cut to `sellable_shares`."""
    if side not in ("buy", "sell"):
        raise ValueError("side must be buy or sell")
    try:
        shares = sh.quantity(shares)
    except ValueError:
        shares = Decimal(0)
    if shares <= 0:
        raise ValueError("shares must be a number above zero (a whole number, or one with at most 6 decimals)")
    try:
        price = exact(price, "price")                   # a float is a TypeError (it comes in through a door)
    except ValueError:                                  # a bool, a string, NaN: the sentence this function always said
        raise ValueError("price must be a finite number above zero") from None
    if price <= 0:
        raise ValueError("price must be a finite number above zero")
    if type(today) is not date:
        raise ValueError("today must be a date")
    if not isinstance(symbol, str):
        raise ValueError("the symbol must be text, like 'MSFT'")
    symbol = check_symbol(symbol.strip().lstrip("$"))
    amount = trade_amount(shares, price)
    if amount <= 0:
        raise ValueError("the trade must move at least one cent")
    held = {p.symbol: p.model_copy() for p in portfolio.positions}
    if side == "buy":
        if portfolio.cash_avail - amount < 0:
            raise ValueError(f"cash ${portfolio.cash_avail:,.2f} does not cover ${amount:,.2f}")
        _buy_into(held, symbol, shares, price, today, name)
        return _with_positions(portfolio, held, portfolio.cash_avail - amount)
    _sell_from(held, symbol, shares)
    return _with_positions(portfolio, held, portfolio.cash_avail + amount)


def load_portfolio(path: str | Path) -> Portfolio:
    """Read a portfolio JSON file. Raises ValueError, with the reason, if the file is not valid JSON or not a valid portfolio."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: not valid JSON ({error.msg})") from None
    return Portfolio.model_validate(data)
