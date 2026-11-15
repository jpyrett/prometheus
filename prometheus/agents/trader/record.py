"""The shapes the Trader writes: one `Trade` per ticket, and one `Run` per call of the Trader (Trader v1 spec section 14).

A trade starts as a TICKET, status `new`, built by code from a trade decision, the Critic's review and the Advisor's plan (`tickets.py`); placing it makes it `filled`, `active` (a resting limit order) or `rejected`.
The trades live on `order.trades` and the runs on `order.trader_runs` as plain dicts (`model_dump(mode="json")`), the same way as the Advisor's recommendations and the Critic's reviews. The Advisor's records, the
Critic's reviews and the Account Manager's decisions are never touched (rule T10).

    Trade   trade_id (T1, or T1-F1 for a sale that funds T1), kind (recommendation | funding), the ids that join it to the record, the review and (funding) the purchase it pays for; symbol, side,
            shares_approved (what the decision allowed) and shares (what is to be traded), the order (type and limit price), the reference price and the limits code computed, status, reject_code and reason,
            who placed it, the model's note, when it was placed and in which mode, the fill, whether it was booked in the account book, and (backtest) the outcome
    Fill    price, shares, amount, commission, the date, where the price came from (quote | bar_open | bar_limit) and the slippage that was applied
    Outcome (backtest only) how a filled purchase went through the later bars: status, exit, gain, return, and the same dates against the benchmark
    Run     one call of the Trader: the model and its temperature, the mode, the tool calls and the tokens, how many trades the model placed and how many code did, the summary and whose it is, the account before and
            after, the price order, the funding answer, whether it was booked, what was dropped and what went wrong

Every model forbids unknown fields: a typo in the code that builds a trade fails when the trade is built. The ONLY places a model's words reach these records are listed in `MODEL_FIELDS`, and a test asserts it
(rule T8). A `rejected` trade carries a code from `REJECT_CODES`, a closed list: the model cannot invent one and neither can the code. A trade's `symbol` is a ticker (`is_ticker`: `check_symbol`'s pattern, already in
capitals): the symbol is the one piece of the order the model is shown as text (`verify_request`), so it must never be a sentence. A decision whose symbol is not a ticker gets no ticket (`tickets.build_tickets`).
"""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from prometheus.common.work_order import TRADE_KINDS, TRADE_SIDES, TRADE_STATUSES
from prometheus.rag.loaders import check_symbol

MODES = ("paper", "backtest")
KINDS, SIDES, STATUSES = TRADE_KINDS, TRADE_SIDES, TRADE_STATUSES   # the work order owns these lists (its output guard `verify_trades` reads them); the Literals below must say the same, and a test asserts it
ORDER_TYPES = ("market", "limit")
PLACED_BY = ("model", "code")
PRICE_SOURCES = ("quote", "bar_open", "bar_limit")
OUTCOME_STATUSES = ("open", "closed")
EXIT_REASONS = ("stop", "target", "time_stop")
SUMMARY_BY = ("model", "code")
# The ONLY places a model's words reach a trade or a run (spec 14): the note of a trade and the summary of a run. Both are checked by code (T8): a note is cut to `limits.max_note_words`, a summary is kept only when every
# trade it names and every number in it is real, and is otherwise replaced by code's own sentence.
MODEL_FIELDS = ("trades[*].note", "trader_runs[*].summary")
# The closed list of reasons a trade is rejected (spec 5). `reason` is the sentence; `reject_code` is one of these.
REJECT_CODES = ("not_permitted", "approval_expired", "price_unavailable", "price_outside_limits", "limit_not_reached", "insufficient_cash",
                "not_held", "not_sellable_today", "funding_failed", "not_attempted", "account_refused", "internal_error")


def is_ticker(value) -> bool:
    """Whether `value` is a ticker as a trade holds it: text that `check_symbol` accepts (letters, digits, '.' and '-', at most 10) and that is ALREADY in capitals with no space around it. `check_symbol` would clean
    ' pg ' to 'PG'; a trade must hold exactly the symbol its decision holds (the output guard compares them), so the clean form is the only form accepted."""
    if not isinstance(value, str):
        return False
    try:
        return check_symbol(value) == value
    except ValueError:
        return False


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Fill(_Model):
    """One execution: the price and the shares are above zero (an order that fills nothing is not a fill)."""
    price: float = Field(gt=0)                         # per share, 4 places
    shares: int = Field(ge=1)
    amount: float                                      # price x shares, in cents (a purchase costs it, a sale brings it in)
    commission: float = Field(default=0.0, ge=0)
    date: str                                          # YYYY-MM-DD: the day of the fill (paper: the Eastern date of the run; backtest: the bar's date)
    price_source: Literal["quote", "bar_open", "bar_limit"]
    slippage_bps: float = Field(default=0.0, ge=0)     # what was added to a buy's price and taken off a sale's


class Outcome(_Model):
    """How a filled purchase went, followed through the stored bars after its entry (backtest mode only: `simulator.follow`; spec section 10). Never booked in the account book."""
    status: Literal["open", "closed"]
    exit_reason: Literal["stop", "target", "time_stop"] | None = None
    exit_date: str | None = None
    exit_price: float | None = None
    sessions_held: int | None = None
    realized_gain: float | None = None                 # closed: (exit price - fill price) x shares
    return_pct: float | None = None                    # closed: against the fill price; open: against the last close
    last_date: str | None = None                       # open (or the last bar read): the date and the close it is marked at
    last_close: float | None = None
    unrealized_gain: float | None = None
    benchmark_return_pct: float | None = None          # the benchmark over the same dates; None when its bars are not stored
    excess_return_pct: float | None = None


class Trade(_Model):
    trade_id: str = Field(min_length=1)
    kind: Literal["recommendation", "funding"]
    record_id: str | None = None                       # the recommendation this trade belongs to (a funding sale carries the record_id of the purchase it pays for)
    review_id: str | None = None
    funds_for: str | None = None                       # a funding sale: the trade_id of the purchase it pays for
    symbol: str
    side: Literal["buy", "sell"]
    shares_approved: int = Field(ge=0)                 # what the Account Manager's decision allowed; the Trader never trades more
    shares: int = Field(ge=0)                          # what the ticket is for: a sale is cut to the shares sellable today, a purchase to the cash when it is placed
    order_type: Literal["market", "limit"] | None = None
    limit_price: float | None = None
    ref_price: float | None = None                     # the price the Critic judged (its deal price), else the Advisor's entry price
    limit_low: float | None = None                     # the range code computed: no order of any type fills outside it
    limit_high: float | None = None
    status: Literal["new", "active", "filled", "rejected"]
    reject_code: str | None = None
    reason: str = ""
    placed_by: Literal["model", "code"] | None = None
    note: str = ""                                     # the model's words (cut by code): one of the two fields in MODEL_FIELDS
    placed_utc: str | None = None
    mode: Literal["paper", "backtest"]
    fill: Fill | None = None
    booked: bool = False                               # the fill was saved in the Postgres account book
    outcome: Outcome | None = None

    @field_validator("symbol")
    @classmethod
    def _the_symbol_is_a_ticker(cls, value: str) -> str:
        if not is_ticker(value):
            raise ValueError(f"a trade's symbol must be a ticker in capitals (letters, digits, '.' and '-', at most 10), not {value[:20]!r}")
        return value

    @model_validator(mode="after")
    def _a_rejection_has_a_known_code(self):
        if self.status == "rejected" and self.reject_code not in REJECT_CODES:
            raise ValueError(f"a rejected trade needs a reject_code from the closed list ({', '.join(REJECT_CODES)}), not {self.reject_code!r}")
        return self


class Snapshot(_Model):
    """The account at one moment: the cash and each position, as the Trader saw it before the run and as it stood after."""
    cash: float
    positions: list[dict[str, Any]] = Field(default_factory=list)   # {symbol, shares, cost_basis, acquired}


class FundingRun(_Model):
    """What the run decided about a funding plan: its status, whether the user said yes to the sells, and a sentence."""
    status: str | None = None                          # the RaiseFunds status ("plan", "funded", ...), None without a plan
    approved: bool | None = None                       # None: nobody was asked (no plan, or the cash covers it)
    note: str = ""


class Run(_Model):
    """One call of the Trader (an entry of `order.trader_runs`). A run that was refused at the first step (PAPER_TRADING off, a mode that is not allowed, an order not decided) still writes one, with `errors`."""
    model: str | None = None                           # the model's id; None when no model ran
    temperature: float = 0.0
    mode: str                                          # str, not a Literal: a refused run records the mode it was asked for ("live")
    now_utc: str
    use_model: bool = True
    turns: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)   # {name, args, outcome}
    placed_by_model: int = 0
    placed_by_code: int = 0
    summary: str = ""                                  # the model's words when code kept them, else code's own sentence: the other field in MODEL_FIELDS
    summary_by: Literal["model", "code"] = "code"
    portfolio_before: Snapshot | None = None
    portfolio_after: Snapshot | None = None
    price_order: str | None = None                     # the request_id of the quote or bars order the Account Manager made
    funding: FundingRun | None = None
    booked: bool = False
    config_version: str = ""
    dropped: list[str] = Field(default_factory=list)   # what code threw away of the model's output, and why
    errors: list[str] = Field(default_factory=list)
