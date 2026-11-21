"""Stock splits: the stored split table of a symbol, and what a position does when it crosses one (design/proving_ground_spec.md P10; design choices entry 126).

    table = load("NFLX")                                         # a `SplitTable`, or None when no table is stored (or it cannot be read): then the suspect rule decides
    table.factor_between(date(2025, 11, 14), date(2025, 11, 20))  # Decimal('10')   the shares that one share of the 14th has become by the 20th
    shares, prices = table.adjust(Decimal(8), {"stop": Decimal("1050.0000"), "target": None}, date(2025, 11, 17))     # (Decimal('80'), {"stop": Decimal('105.0000'), "target": None})
    adjusted = to_basis(bars, table, date(2025, 11, 14))         # the bars of a symbol, from the stored (UNADJUSTED) prices into the share basis of the 14th, the day a plan was made
    moved = suspect_move(bars, date(2025, 11, 14), date(2025, 12, 5), Decimal("0.40"))     # (date(2025, 11, 17), Decimal('-0.9...')): a close-to-open move that looks like a split

THE STORED BARS ARE UNADJUSTED. A split drops the price on its execution date (10 for 1: a tenth) and multiplies the shares, so a position that crosses one would read as a loss and hit its stop. The table says which splits happened:
a split executes on `execution_date` (the first day the stock trades at the new price) and one old share becomes `split_to / split_from` new ones (a 10-for-1 is `split_from` 1, `split_to` 10; a 1-for-10 reverse split is
`split_from` 10, `split_to` 1). `adjust` is what a position does ON that date: its shares are multiplied (a fraction below the six places of a share quantity is dropped, never rounded up: cash in lieu) and its plan prices and cost
basis are divided (the price door: four places, half up). `to_basis` is the same thing seen from the bars: a bar on or after a split is put back into the share basis of the day a plan was made (its prices multiplied by the factor of every split
since that day), so that the entry price, the stop and the target of a plan, which were made before the split, can be compared with it: `simulator.follow` is reused unchanged on those bars.

THE FILE (`data/historical/splits_v1/<SYMBOL>.json`, written by `DataToolbox.fetch_splits` from Massive's reference splits; read only here). `split_from` and `split_to` are JSON floats (1.0, 4.0), read by their digits:

    {"symbol": "AAPL", "source": "Massive /stocks/v1/splits", "fetched_utc": "2026-11-20T18:06:00Z",
     "splits": [{"ticker": "AAPL", "execution_date": "2020-08-31", "split_from": 1.0, "split_to": 4.0, "adjustment_type": "forward_split", "id": "P5cb1..."}]}

`adjustment_type` is `forward_split`, `reverse_split` or `stock_dividend` (a 5% stock dividend is 20 to 21: the same arithmetic). An unknown key is ignored. A MISSING FILE means "no table", and so does one that cannot be read (or is for another
symbol); the caller then applies THE SUSPECT RULE: a symbol with no table gets no outcome across a close-to-open move of more than `split_suspect_pct` (40%) and says why (`suspect_move`). A table with an empty list of splits means "read, and the
symbol did not split": it is trusted.
"""
import json
from datetime import date
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from prometheus.common import shares as sh
from prometheus.common.money import EXACT, digits, price as price_door, price_scale
from prometheus.rag.config import REPO_ROOT
from prometheus.rag.loaders import check_symbol

DEFAULT_ROOT = REPO_ROOT / "data" / "historical" / "splits_v1"


def _ratio(value) -> Decimal:
    """A number of a split ("split_from": 1, "split_to": 10): a number above zero, read from its written digits."""
    d = digits(value)
    if d <= 0:
        raise ValueError(f"a split ratio is a number above zero, not {value!r}")
    return d


Ratio = Annotated[Decimal, BeforeValidator(_ratio)]


class Split(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    execution_date: date
    split_from: Ratio
    split_to: Ratio

    @property
    def factor(self) -> Decimal:
        """The shares one old share becomes: 10 for a 10-for-1 split, one half for a 1-for-2 reverse split."""
        return EXACT.divide(self.split_to, self.split_from)


class SplitTable(BaseModel):
    model_config = ConfigDict(extra="ignore")
    symbol: str
    fetched_utc: str | None = None
    splits: list[Split] = Field(default_factory=list)

    @model_validator(mode="after")
    def _in_order_and_once(self):
        days = [s.execution_date for s in self.splits]
        if days != sorted(set(days)):
            raise ValueError("the splits of a table are in date order and each date once")
        return self

    def on(self, day: date) -> list[Split]:
        """The splits that execute exactly on `day`."""
        return [s for s in self.splits if s.execution_date == day]

    def factor_between(self, after: date, through: date) -> Decimal:
        """The shares that one share held at the close of `after` has become by the close of `through`: the product of the factors of the splits executing after `after` and on or before `through` (1 when there is none)."""
        factor = Decimal(1)
        for split in self.splits:
            if after < split.execution_date <= through:
                factor = EXACT.multiply(factor, split.factor)
        return factor

    def adjust(self, shares, prices: dict[str, Decimal | None], day: date) -> tuple[Decimal, dict[str, Decimal | None]]:
        """A position on the day of a split: (its shares multiplied by the factor, its prices divided by it). `shares` is a quantity; `prices` is {name: price or None} (a plan's stop and target, a cost basis). A day with no split
        gives the same shares and the same prices. A fraction of a share below the six places of a quantity is dropped (never rounded up); a price is rounded once, by the price door (four places, half up)."""
        count = sh.quantity(shares)
        factor = Decimal(1)
        for split in self.on(day):
            factor = EXACT.multiply(factor, split.factor)
        if factor == 1:
            return count, dict(prices)
        return sh.quantity(EXACT.multiply(count, factor).quantize(sh.STEP, rounding=ROUND_DOWN)), {k: None if v is None else price_scale(EXACT.divide(v, factor)) for k, v in prices.items()}


def read(symbol: str, root: str | Path | None = None) -> tuple[SplitTable | None, str]:
    """(the symbol's split table, "") or (None, why there is none): no file is stored, or the file cannot be read (not JSON, not the shape above). Never raises for a file; a symbol that is not a ticker is a ValueError."""
    symbol = check_symbol(symbol)
    path = Path(root or DEFAULT_ROOT) / f"{symbol}.json"
    if not path.is_file():
        return None, f"no split table is stored for {symbol}"
    try:
        table = SplitTable.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as error:       # a JSON error and a pydantic error are both ValueErrors
        return None, f"the split table of {symbol} could not be read ({type(error).__name__})"
    if table.symbol != symbol:
        return None, f"the split table of {symbol} is for {table.symbol}"
    return table, ""


def load(symbol: str, root: str | Path | None = None) -> SplitTable | None:
    """The symbol's split table, or None when there is none (see `read`)."""
    return read(symbol, root)[0]


def to_basis(bars: list[dict], table: SplitTable | None, basis_day: date) -> list[dict]:
    """The stored bars of a symbol in the share basis of `basis_day` (the day a plan was made, or the day a position was bought): each bar after a split has its open, high, low and close multiplied by the factor of every split that
    executed after `basis_day` and on or before the bar's own date (a split on or before `basis_day` changes nothing: that day's price is already the new one). The prices come out through the price door (four places, half up), as
    Decimals; a bar before every split, and every bar when there is no table or no split, is returned as it is. Volume is left as it was."""
    if table is None or not table.splits:
        return list(bars)
    out = []
    for bar in bars:
        day = bar.get("date") if isinstance(bar, dict) else None
        factor = table.factor_between(basis_day, date.fromisoformat(day)) if isinstance(day, str) else Decimal(1)
        if factor == 1:
            out.append(bar)
        else:
            out.append({**bar, **{k: price_door(EXACT.multiply(price_door(bar[k]), factor)) for k in ("open", "high", "low", "close") if bar.get(k) is not None}})
    return out


def suspect_move(bars: list[dict], first_day: date, last_day: date, pct: Decimal) -> tuple[date, Decimal] | None:
    """THE SUSPECT RULE: the first close-to-open move, from one stored bar to the next, of more than `pct` (0.40 is 40%) with the later bar between `first_day` (excluded: the close of that day is the price the plan was made at) and `last_day`
    (included): (the date of the later bar, the move as a Decimal fraction, 6 places). None when there is none. A move that big in one night is a split or something no plan was made for, and a stored bar cannot tell which."""
    series = sorted((b for b in bars if isinstance(b, dict) and isinstance(b.get("date"), str) and b.get("open") is not None and b.get("close") is not None), key=lambda b: b["date"])
    for before, after in zip(series, series[1:]):
        day = date.fromisoformat(after["date"])
        if not first_day < day <= last_day:
            continue
        move = EXACT.divide(price_door(after["open"]), price_door(before["close"])) - 1
        if abs(move) > pct:
            return day, move.quantize(Decimal("0.000001"))
    return None
