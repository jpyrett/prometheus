"""A user's portfolio: what the Account Manager reads when it has to raise funds (design choices entry 42).

    portfolio = load_portfolio("configs/portfolios/jacob_pyrett.json")
    portfolio.position("TSLA").shares                       # 123
    portfolio.sellable_shares("TSLA", today=date.today())   # shares that may be sold today: not bought today (the design's T+1 holding rule)

The JSON files are still what the desk loads by default. The account book (design choices entry 72) loads this same class from Postgres. Nothing here calls an API or an LLM.

    {"schema_version": 1, "user": "Jacob Pyrett", "currency": "USD", "cash_avail": "51.23",
     "positions": [{"symbol": "MSFT", "name": "Microsoft Corp", "shares": 50, "cost_basis": 150.0, "acquired": "2024-03-15"}, ...]}

`cash_avail` is the FREE SETTLED CASH: the money that can buy today. RaiseFunds spends it first, and only sells what is still missing. It may be a number or a string of a number
("51.23"), as in the sample file; it must be at least 0 and finite. `cost_basis` is the average price paid per share. `acquired` is the date of the latest purchase of the position (left out: all of it is sellable).
"""
import json
import math
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

from prometheus.rag.loaders import check_symbol


class Position(BaseModel):
    symbol: str
    name: str | None = None
    shares: int = Field(gt=0)
    cost_basis: float = Field(gt=0)              # average price paid per share
    acquired: date | None = None                 # when the latest shares were bought; None means long ago
    recent_shares: int | None = Field(default=None, gt=0)  # how many of the shares were bought on `acquired` (the T+1 lock applies to these only); None: all of the position

    @field_validator("symbol")
    @classmethod
    def _symbol(cls, value: str) -> str:
        return check_symbol(value.strip().lstrip("$"))


class Portfolio(BaseModel):
    schema_version: int = 1
    user: str
    currency: str = "USD"
    cash_avail: float = Field(default=0.0, ge=0, allow_inf_nan=False)  # free settled cash, in `currency`: what can buy today. Sale proceeds that have not settled (T+1) are not in it
    positions: list[Position] = Field(default_factory=list)
    note: str | None = None

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

    def sellable_shares(self, symbol: str, today: date | None = None) -> int:
        """The shares of `symbol` that may be sold today. The design's T+1 holding rule: shares bought today are not sellable until tomorrow. If the position says how many of its
        shares were bought on `acquired` (`recent_shares`), only those are locked and the older ones are sellable; without it the whole position is locked. A position whose purchase
        date is not given is all sellable."""
        position = self.position(symbol)
        if position is None:
            return 0
        today = today or date.today()
        if position.acquired is None or position.acquired < today:
            return position.shares
        return 0 if position.recent_shares is None else max(0, position.shares - position.recent_shares)

    def market_value(self, prices: dict[str, float]) -> float:
        return sum(p.shares * prices[p.symbol] for p in self.positions)


def trade_amount(shares: int, price: float) -> float:
    """What `shares` at `price` cost, or bring in, in cents: the way the account book rounds it (`round(price x shares, 2)` half up, on the price as it is written: 500.125 x 1 is 500.13, where Python's
    own `round` gives 500.12). `apply_fill`, the Trader's paper broker and a fill's `amount` all use it, so that the portfolio in memory, the fill and the booked cash never differ by a cent."""
    return float((Decimal(str(price)) * shares).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def affordable_shares(cash: float, price: float, shares: int, commission: float = 0.0) -> int:
    """How many of `shares` the cash pays for at `price`, the way the account does the sum: the most shares `n` (0 to `shares`) with `round(cash - trade_amount(n, price) - commission, 2) >= 0` (an amount is rounded half up to
    the cent, `trade_amount`, and the account refuses a purchase the cash does not cover to the cent). 0 when not even one share is affordable. This is `orders.cash_policy: "reduce"` of the Trader: buying fewer shares than
    approved is inside the permission, buying more never is. The Account Manager's `approve_trade` is the only caller (design choices entry 81: the money is the Account Manager's, the Trader never works it out); it lived in the
    Trader's `simulator.py` until then."""
    usable = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0
    if not usable(price) or not isinstance(shares, int) or isinstance(shares, bool) or shares < 1 or not usable(cash):
        return 0
    n = min(shares, int(max(cash - commission, 0.0) // price) + 1)       # the quotient is exact up to float error: start one above it and step down to the first that fits
    while n > 0 and round(cash - trade_amount(n, price) - commission, 2) < 0:
        n -= 1
    return n


def _sell_from(held: dict[str, Position], symbol: str, shares) -> None:
    """Take `shares` of `symbol` out of `held` (a dict of position copies, changed in place); a position sold out completely is removed. The locked shares are sold last: a caller only sells what is sellable,
    so this only keeps `recent_shares` inside the position. Raises ValueError for a sell the portfolio cannot make (a symbol it does not hold, a count that is not a whole number from 1 to the shares held)."""
    position = held.get(symbol)
    if position is None or not isinstance(shares, int) or not 1 <= shares <= position.shares:
        raise ValueError(f"cannot sell {shares} of {symbol}: the portfolio holds {position.shares if position else 0}")
    position.shares -= shares
    if position.recent_shares is not None:
        position.recent_shares = min(position.recent_shares, position.shares) or None
    if position.shares == 0:
        del held[symbol]


def _buy_into(held: dict[str, Position], symbol: str, shares: int, price: float, today: date, name: str | None = None) -> None:
    """Add `shares` of `symbol` bought at `price` on `today` to `held` (a dict of position copies, changed in place): THE rules of the T+1 lock and of the cost basis, in one place (`after_purchase` and
    `apply_fill` both use it). A new position is all bought today, so all of it is locked. A symbol already held gets the weighted average cost basis, `acquired` becomes today, and only the new shares
    are locked (the older ones stay sellable), unless the position was already wholly locked, which it stays. `name` is kept when the position has one and none is given."""
    existing = held.get(symbol)
    if existing is None:
        held[symbol] = Position(symbol=symbol, name=name, shares=shares, cost_basis=price, acquired=today)  # a new position: all of it was bought today, so all of it is locked
        return
    total = existing.shares + shares
    if existing.acquired is not None and existing.acquired >= today:  # it was already bought into today
        locked = None if existing.recent_shares is None else existing.recent_shares + shares  # None: the whole position is locked, and stays so
    else:
        locked = shares  # the older shares stay sellable; only the new ones are locked
    held[symbol] = existing.model_copy(update={"shares": total, "cost_basis": round((existing.shares * existing.cost_basis + shares * price) / total, 4),
                                               "acquired": today, "recent_shares": locked, "name": name or existing.name})


def _with_positions(portfolio: Portfolio, held: dict[str, Position], cash: float) -> Portfolio:
    """A NEW portfolio with these positions (the symbols the portfolio already had keep their place, new ones come last) and this cash."""
    order = [s for s in portfolio.symbols if s in held] + [s for s in held if s not in portfolio.symbols]
    return portfolio.model_copy(update={"positions": [held[s] for s in order], "cash_avail": round(cash, 2)})


def after_purchase(portfolio: Portfolio, target: str, shares: int, price: float, sells: list[tuple[str, int]], cash_after: float, today: date | None = None) -> Portfolio:
    """The portfolio once a plan has been carried out: the `sells` taken out (a position sold out completely is removed), `shares` of `target` bought at `price`, `cash_avail` set to `cash_after`
    (what the plan says is left). The shares bought are marked as bought `today` (`acquired`, `recent_shares`), so that the T+1 lock keeps them from being sold in the next plan of the same day,
    and the older shares of the same stock stay sellable. A target already held gets the weighted average cost basis. Returns a NEW portfolio: the one given is not changed. Raises
    ValueError for a sell the portfolio cannot make (a symbol it does not hold, more shares than held)."""
    today = today or date.today()
    held = {p.symbol: p.model_copy() for p in portfolio.positions}
    for symbol, n in sells:
        _sell_from(held, symbol, n)
    _buy_into(held, target, shares, price, today)
    return _with_positions(portfolio, held, cash_after)


def apply_fill(portfolio: Portfolio, symbol: str, side: str, shares: int, price: float, today: date, name: str | None = None) -> Portfolio:
    """The portfolio after ONE filled trade (Trader v1, design/trader_v1_final_spec.md section 13): the Account Manager's `notify_trades` and the Trader's paper broker both call it, so they cannot disagree.

    A buy costs `trade_amount(shares, price)` (`round(shares * price, 2)`, half up, as the account book rounds), which the cash must cover; the position follows the rules `after_purchase` has (the average cost basis, `acquired`, `recent_shares` for the T+1 lock: the same code).
    A sell needs the shares to be held; the proceeds, `trade_amount(shares, price)`, are added to the cash, the position keeps `recent_shares` inside itself, and a position sold out is removed. A trade must move
    at least one cent. `name` names a position that is new (or has no name). Returns a NEW portfolio; the one given is not changed. Raises ValueError, with the reason, for anything it cannot apply:
    a side that is not buy or sell, a count that is not a whole number of at least 1, a price that is not a finite number above zero, a date that is not a date, a symbol that is not a ticker, a purchase the cash does not cover, a sale of shares that are not held.
    It does not check the T+1 lock of a sale: the Trader's tickets are already cut to `sellable_shares`."""
    if side not in ("buy", "sell"):
        raise ValueError("side must be buy or sell")
    if isinstance(shares, bool) or not isinstance(shares, int) or shares < 1:
        raise ValueError("shares must be a whole number of at least 1")
    if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
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
        if round(portfolio.cash_avail - amount, 2) < 0:
            raise ValueError(f"cash ${portfolio.cash_avail:,.2f} does not cover ${amount:,.2f}")
        _buy_into(held, symbol, shares, float(price), today, name)
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
