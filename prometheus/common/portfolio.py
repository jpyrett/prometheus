"""A user's portfolio: what the Account Manager reads when it has to raise funds (design choices entry 42).

    portfolio = load_portfolio("configs/portfolios/jacob_pyrett.json")
    portfolio.position("TSLA").shares                       # 123
    portfolio.sellable_shares("TSLA", today=date.today())   # shares that may be sold today: not bought today (the design's T+1 holding rule)

It is a static JSON file for now. The week 2 design keeps the real portfolio in Postgres (ACID, row locks); until the Account Manager has that database this file stands in for
it, and the same class is what the database rows would be loaded into. Nothing here calls an API or an LLM.

    {"schema_version": 1, "user": "Jacob Pyrett", "currency": "USD", "cash_avail": "51.23",
     "positions": [{"symbol": "MSFT", "name": "Microsoft Corp", "shares": 50, "cost_basis": 150.0, "acquired": "2024-03-15"}, ...]}

`cash_avail` is the FREE SETTLED CASH: the money that can buy today. RaiseFunds spends it first, and only sells what is still missing. It may be a number or a string of a number
("51.23"), as in the sample file; it must be at least 0 and finite. `cost_basis` is the average price paid per share. `acquired` is the date of the latest purchase of the position (left out: all of it is sellable).
"""
import json
from datetime import date
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


def after_purchase(portfolio: Portfolio, target: str, shares: int, price: float, sells: list[tuple[str, int]], cash_after: float, today: date | None = None) -> Portfolio:
    """The portfolio once a plan has been carried out: the `sells` taken out (a position sold out completely is removed), `shares` of `target` bought at `price`, `cash_avail` set to `cash_after`
    (what the plan says is left). The shares bought are marked as bought `today` (`acquired`, `recent_shares`), so that the T+1 lock keeps them from being sold in the next plan of the same day,
    and the older shares of the same stock stay sellable. A target already held gets the weighted average cost basis. Returns a NEW portfolio: the one given is not changed. Raises
    ValueError for a sell the portfolio cannot make (a symbol it does not hold, more shares than held)."""
    today = today or date.today()
    held = {p.symbol: p.model_copy() for p in portfolio.positions}
    for symbol, n in sells:
        position = held.get(symbol)
        if position is None or not isinstance(n, int) or not 1 <= n <= position.shares:
            raise ValueError(f"cannot sell {n} of {symbol}: the portfolio holds {position.shares if position else 0}")
        position.shares -= n
        if position.recent_shares is not None:  # the locked shares are sold last: a plan only sells what is sellable, so this only keeps the number inside the position
            position.recent_shares = min(position.recent_shares, position.shares) or None
        if position.shares == 0:
            del held[symbol]
    existing = held.get(target)
    if existing is None:
        held[target] = Position(symbol=target, shares=shares, cost_basis=price, acquired=today)  # a new position: all of it was bought today, so all of it is locked
    else:
        total = existing.shares + shares
        if existing.acquired is not None and existing.acquired >= today:  # it was already bought into today
            locked = None if existing.recent_shares is None else existing.recent_shares + shares  # None: the whole position is locked, and stays so
        else:
            locked = shares  # the older shares stay sellable; only the new ones are locked
        held[target] = existing.model_copy(update={"shares": total, "cost_basis": round((existing.shares * existing.cost_basis + shares * price) / total, 4),
                                                   "acquired": today, "recent_shares": locked})
    order = [s for s in portfolio.symbols if s in held] + [s for s in held if s not in portfolio.symbols]
    return portfolio.model_copy(update={"positions": [held[s] for s in order], "cash_avail": round(cash_after, 2)})


def load_portfolio(path: str | Path) -> Portfolio:
    """Read a portfolio JSON file. Raises ValueError, with the reason, if the file is not valid JSON or not a valid portfolio."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: not valid JSON ({error.msg})") from None
    return Portfolio.model_validate(data)
