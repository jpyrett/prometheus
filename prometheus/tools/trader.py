"""The Trader's tools: what its model may ask for, and nothing else (Trader v1 spec section 8, rules T1 to T3 and T9).

Each tool is a plain method of `TraderTools` with type hints and a docstring written for the model, as `prometheus/tools/data_clerk.py` does: LangChain reads the name, the docstring and the typed arguments to build the tool
the model sees.

    TRADER_TOOLS          the allow-list of FOUR: verify_request, place_order, check_trade_status, notify_account_manager
    TraderTools(broker, notifier)   the tools themselves, over the run's `PaperBroker` and the run's one-shot `notifier` (execute.py: it closes the run and reads the account back)
    .langchain_tools()    the LangChain tools
    Budget, budgeted      what one run may use (a count per tool, `place_order` once per trade), and the tool behind it
    describe_tool_error   the message the model gets when a tool refuses or fails
    ToolInputError        arguments the model can fix and send again
    ToolLimitError        a limit of the budget was reached

WHAT KEEPS THE MODEL INSIDE ITS PERMISSION (the tools are the safeguard; the temperature is not):
  T1  `place_order` goes through the broker, which asks `work_order.verify_trade` again for every trade it places;
  T2  no tool takes a symbol, a side or a number of shares, and no tool makes a trade: `place_order` takes the `trade_id` that code issued, an order type and a limit price that must be inside the range code
      computed. An argument that is not in the signature is REFUSED, not ignored, so a model that tries to add `symbol=` or `shares=` is told so;
  T3  a trade is placed once: the broker refuses a trade that is not `new`, and the budget allows `place_order` once per trade_id;
  T9  every result holds symbols, numbers and words from closed lists, all made by code: no thesis, no quote, no company name. A rejected trade is shown as its `reject_code`, a word of the closed list, and NEVER as its
      `reason`: that sentence is built from `verify_trade`, which repeats two values of the order (`will_trade`, `valid_until_utc`) that, on an order read from a file, could be any text. The sentence stays on the
      record and in the report. (A refusal message is a sentence too: code writes it from the trade id, the limits and the model's own argument cut short.)

THE HANDSHAKE IS INSIDE `place_order` (design choices entry 81). The Account Manager approves and books each trade as it is placed (`PaperBroker._fill`), so the tool needs nothing new to be safe, and the model cannot skip or
reorder it. What it reads of the handshake is two things code made: `account`, one word of a closed list (`booked`, `applied`, `refused`, `unconfirmed`: what the Account Manager did with the fill) and `cash_left`, a number
(the cash the Account Manager says the account has after it). A refusal reaches it as a reject_code (`insufficient_cash`, `account_refused`), never as the book's sentence: no free text from the book reaches the model.
`notify_account_manager` no longer tells the Account Manager anything (each fill is already told): it closes the run and reads the account back.

DECIMAL MONEY (design choices entry 112): the prices, shares, cash and amounts the tools show are `Decimal`s in the dicts they return and are written to the model as JSON NUMBERS by the one encoder (`tool_content`: `money.dumps`; a Decimal is never a string). The
model's `limit_price` is the number it wrote in its tool call (a float at that door): `place_order` turns it into a Decimal through the price door before the broker sees it; a bool is refused.
"""
from decimal import Decimal
from typing import Callable

from langchain_core.tools import BaseTool, StructuredTool, tool
from pydantic import ValidationError

from prometheus.agents.trader.broker import PaperBroker, Refused
from prometheus.agents.trader.record import Trade
from prometheus.agents.trader.tickets import shown_range
from prometheus.common.money import dumps, price as price_door

TRADER_TOOLS = ("verify_request", "place_order", "check_trade_status", "notify_account_manager")
VERIFY_CALLS = 2            # verify_request: once to read the tickets, once more to read what became of them
NOTIFY_CALLS = 2            # notify_account_manager: once, and one more for a call that was refused because a ticket was still unplaced
STATUS_EXTRA = 2            # check_trade_status: one call for each ticket, and two more
REFUSED_ATTEMPTS = 3        # a place_order the broker refused (a bad price, a purchase before a sale) may be corrected this many times per trade_id; a placing that happened is final


class ToolInputError(ValueError):
    """The model sent arguments that are not allowed. It can fix them and call again."""


class ToolLimitError(Exception):
    """A limit on this run's tool use was reached. Raised by the budget, not by a tool; the message tells the model to move on."""


def describe_tool_error(error: Exception) -> str:
    """The message the model gets when a tool refuses or fails. Short and factual, and it says what to do next. Everything in it is code's own sentence or the model's own argument echoed back (cut short)."""
    if isinstance(error, ValidationError):
        text = "; ".join(f"{'.'.join(str(x) for x in e['loc']) or 'arguments'}: {e['msg']}" for e in error.errors())[:300]
        return f"Invalid arguments: {text}. Fix the arguments and call the tool again."
    text = " ".join(str(error).split())[:300]
    if isinstance(error, ToolLimitError):
        return f"Limit reached: {text}. Do not call this tool for this again. Move on, or answer now with what you have."
    if isinstance(error, (ToolInputError, Refused)):
        return f"Invalid arguments: {text}. Fix the arguments and call the tool again."
    return f"The tool failed ({type(error).__name__}): {text}. Do not repeat the same call."


class Budget:
    """What one run may use: `verify_request` twice, `check_trade_status` once per ticket and two more, `notify_account_manager` twice, and `place_order` once per `trade_id` (a call the broker REFUSES does not use the
    trade's one placing, so the model can correct a bad price, but at most `REFUSED_ATTEMPTS` times). `check` raises ToolLimitError when a call is not allowed and counts the attempt in `stopped`."""

    def __init__(self, ticket_ids: list[str], calls: dict[str, int] | None = None):
        self.limits = {"verify_request": VERIFY_CALLS, "check_trade_status": len(ticket_ids) + STATUS_EXTRA, "notify_account_manager": NOTIFY_CALLS, **(calls or {})}
        self.used: dict[str, int] = {}
        self.placed: set[str] = set()                   # trade ids a place_order went through for (placed, rested or rejected: all final for the model)
        self.refused: dict[str, int] = {}
        self.stopped: list[str] = []                    # every call a limit stopped, for the run

    def check(self, name: str, args: dict) -> None:
        try:
            if name == "place_order":
                key = str(args.get("trade_id", ""))
                if key in self.placed:
                    raise ToolLimitError(f"place_order was already used for {key[:20]}: a trade is placed once")
                if self.refused.get(key, 0) >= REFUSED_ATTEMPTS:
                    raise ToolLimitError(f"place_order for {key[:20]} was refused {REFUSED_ATTEMPTS} times: leave it, the program places it")
                return
            self.used[name] = self.used.get(name, 0) + 1
            if self.used[name] > self.limits.get(name, 0):
                raise ToolLimitError(f"{name} may be used at most {self.limits.get(name, 0)} times in a run and it already was")
        except ToolLimitError as error:
            self.stopped.append(f"{name}: {error}")
            raise

    def observe(self, name: str, args: dict, refused: bool) -> None:
        """Note the result of a call that was allowed: a place_order that went through uses the trade's one placing, one that was refused counts as an attempt."""
        if name == "place_order":
            key = str(args.get("trade_id", ""))
            if refused:
                self.refused[key] = self.refused.get(key, 0) + 1
            else:
                self.placed.add(key)


def budgeted(wrapped: StructuredTool, budget: Budget) -> StructuredTool:
    """The same tool (name, description, arguments) behind the budget: the call is checked and counted first, and what it did (a refusal or a placing) is noted after."""
    def run(**kwargs):
        budget.check(wrapped.name, kwargs)
        try:
            output = wrapped.func(**kwargs)
        except (ToolInputError, Refused):
            budget.observe(wrapped.name, kwargs, refused=True)
            raise
        budget.observe(wrapped.name, kwargs, refused=False)
        return output

    return StructuredTool(name=wrapped.name, description=wrapped.description, args_schema=wrapped.args_schema, func=run)


def call_tool(tools: dict[str, StructuredTool], name: str, args) -> tuple[str, str]:
    """ONE tool call of the model, as the loop's tools node makes it: (outcome, what the model reads). The outcome is `ok`, `invalid` (arguments the model can fix, a trade it may not place), `limit` (the budget) or
    `error` (a bug or a failure in the tool). It never raises: every failure is a message. An argument the tool does not have is REFUSED here, not dropped (the schema alone would drop it silently), so that a model that
    adds `symbol=`, `side=` or `shares=` is told it has no such thing and nothing is traded (rule T2). The tool's own checks, and the broker's, are behind it."""
    chosen = tools.get(name) if isinstance(name, str) else None
    if chosen is None:
        return "invalid", f"Invalid arguments: there is no tool {str(name)[:30]!r}: your tools are {', '.join(TRADER_TOOLS)}."
    args = args if isinstance(args, dict) else {}
    unknown = sorted(str(k) for k in set(args) - set(chosen.args))
    if unknown:
        return "invalid", describe_tool_error(ToolInputError(f"{name} has no argument {', '.join(repr(u) for u in unknown)}: its arguments are {', '.join(sorted(chosen.args)) or 'none'}. A symbol, a side and a number of shares are not yours to set"))
    try:
        return "ok", tool_content(chosen.invoke(args))
    except ToolLimitError as error:
        return "limit", describe_tool_error(error)
    except (ToolInputError, Refused, ValidationError) as error:
        return "invalid", describe_tool_error(error)
    except Exception as error:   # a failure inside a tool must not end the run: the model reads that it failed and the program places what is left
        return "error", describe_tool_error(error)


def loggable(args) -> dict:
    """The arguments of a call as the run records them (rule T8: the model's words reach a record in two places only, a trade's note and the run's summary): the values of the arguments the tools have, cut short, the
    `note` as its number of words, and an argument the tool does not have as `(refused)`."""
    known = {"trade_id", "order_type", "limit_price"}
    out: dict = {}
    for key, value in (args if isinstance(args, dict) else {}).items():
        key = str(key)[:30]
        if key == "note":
            out["note_words"] = len(value.split()) if isinstance(value, str) else 0
        elif key in known:
            out[key] = value[:40] if isinstance(value, str) else value if isinstance(value, (int, float, bool)) or value is None else "(refused)"
        else:
            out[key] = "(refused)"
    return out


class TraderTools:
    def __init__(self, broker: PaperBroker, notifier: Callable[[], dict]):
        """broker: the run's PaperBroker (it holds the trades and the one snapshot of prices, and does the handshake with the Account Manager). notifier: a callable with a boolean `done`, which closes the run and reads the
        account back from the Account Manager the first time it is called and gives the first answer again afterwards (the Trader's step 8, see execute.py)."""
        self.broker, self.notifier = broker, notifier

    # ---------------------------------------------------------------- views: symbols, numbers and words from closed lists, made by code

    def _view(self, trade: Trade) -> dict:
        out = {"trade_id": trade.trade_id, "symbol": trade.symbol, "side": trade.side, "shares": trade.shares, "status": trade.status}
        if trade.status == "new":
            price = self.broker.venue.price_of(self.broker.prices, trade.symbol)      # the price the model is told a trade stands at: the venue says (the quote for paper, the entry bar's open for a backtest)
            out["market_price"] = price
            out["default_order"] = {"order_type": trade.order_type, "limit_price": trade.limit_price}
            out["limit_range"] = shown_range(trade.limit_low, trade.limit_high) if trade.limit_low is not None and trade.limit_high is not None else None
        elif trade.status == "filled" and trade.fill is not None:
            out.update(fill_price=trade.fill.price, fill_shares=trade.fill.shares, amount=trade.fill.amount)
        elif trade.status == "rejected":
            out["reject_code"] = trade.reject_code           # a word of the closed list; never `reason` (rule T9, see the module text)
        if funded := self._funded(trade):
            out["funded_by"] = funded
        return out

    def _funded(self, trade: Trade) -> list[dict]:
        """The sales that pay for a purchase: symbol and shares while they are waiting, with their status once they were judged. Code's own words and numbers."""
        legs = [t for t in self.broker.trades.values() if t.kind == "funding" and t.funds_for == trade.trade_id]
        return [{"symbol": leg.symbol, "shares": leg.shares} if trade.status == "new" else {"symbol": leg.symbol, "shares": leg.fill.shares if leg.fill else leg.shares, "status": leg.status} for leg in legs]

    # ---------------------------------------------------------------- the four tools

    def verify_request(self) -> dict:
        """List the trades of this work order: the tickets the program made for you. Call it first.

        Each trade has a trade_id, a symbol, a side (buy or sell), a number of shares and a status. A trade whose status is "new" also has a market_price, a default_order (the order that is placed when you give none) and a
        limit_range, the lowest and highest limit price you may use. A trade that is "rejected" is final and carries a reject_code, one word that says why (for example not_permitted, approval_expired, price_unavailable, price_outside_limits, insufficient_cash, account_refused). A purchase that is paid for by selling other holdings lists them in funded_by:
        the program sells them together with the purchase, you do not place them. You cannot make, change or remove a trade.
        """
        broker = self.broker
        cash = broker.cash_now()       # the Account Manager's figure, read now: the Trader keeps none of its own
        return {"mode": broker.mode, "cash": cash, "trades": [self._view(t) for t in broker.trades.values() if t.kind == "recommendation"]}

    def place_order(self, trade_id: str, order_type: str = "", limit_price: float = 0.0, note: str = "") -> dict:      # `limit_price`: the number the model wrote in its tool call (a JSON number is a float here): it becomes a Decimal at the first line below, once
        """Place one trade that is "new". Give its trade_id and nothing else for the default order, which is almost always right.

        A trade is placed once: a second call for the same trade_id is refused. Every sale must be placed before any purchase. The price of a limit order must be inside the limit_range of verify_request; the program
        also checks that the trade is still permitted. The Account Manager approves every trade against the cash and holdings the account has at that moment, and records a fill at once, before the next trade can be placed:
        it can approve fewer shares than the trade has, or none (reject_code insufficient_cash), and a fill the account refuses is rejected (reject_code account_refused). The result says whether the trade was filled, is
        waiting (active) or was rejected (with its reject_code), at what price, and for a fill what the Account Manager did with it (account: booked, applied, refused or unconfirmed) and the cash left in the account.

        Args:
            trade_id: The trade to place, such as "T1", exactly as verify_request lists it. A funding sale cannot be placed by you.
            order_type: "market" or "limit". Leave it empty for the default order.
            limit_price: The limit price, for a limit order, inside the limit_range. 0 means the default price. A market order takes no price.
            note: A few words on why, at most 20. Optional.
        """
        try:
            limit = Decimal(0) if limit_price == 0 and not isinstance(limit_price, bool) else price_door(limit_price)      # the model's number through THE price door (a bool is refused here; four places, half up)
        except ValueError:
            raise Refused("limit_price must be a number (0 for the default)") from None
        trade = self.broker.place(trade_id, order_type, limit, note, placed_by="model")
        funded, told = self._funded(trade), self.broker.answers.get(trade.trade_id)
        handshake = {"account": told["account"], **({"cash_left": told["cash_left"]} if told["cash_left"] is not None else {})} if told else {}      # one word of a closed list and a number: what the Account Manager did, as code wrote it
        return {"trade_id": trade.trade_id, "status": trade.status, "shares": trade.fill.shares if trade.fill else trade.shares, "fill_price": trade.fill.price if trade.fill else None,
                "amount": trade.fill.amount if trade.fill else None, "reject_code": trade.reject_code, **handshake, **({"funded_by": funded} if funded else {})}

    def check_trade_status(self, trade_id: str = "") -> dict:
        """Read the status of one trade, or of every trade when trade_id is empty: new, active (a limit order that is waiting), filled or rejected (with its reject_code). It only reads: it does not look at the market again.

        Args:
            trade_id: The trade to read, such as "T1". Empty means every trade.
        """
        if trade_id:
            trade = self.broker.trades.get(trade_id)
            if trade is None or trade.kind != "recommendation":
                raise ToolInputError(f"there is no trade {str(trade_id)[:30]!r} on this work order: use a trade_id from verify_request")
            return {"trades": [self._status(trade)]}
        return {"trades": [self._status(t) for t in self.broker.trades.values() if t.kind == "recommendation"]}

    def _status(self, trade: Trade) -> dict:
        out = {"trade_id": trade.trade_id, "status": trade.status}
        if trade.status == "filled" and trade.fill is not None:
            out.update(fill_price=trade.fill.price, fill_shares=trade.fill.shares)
        elif trade.status == "rejected":
            out["reject_code"] = trade.reject_code           # a word of the closed list; never `reason` (rule T9, see the module text)
        elif trade.status == "active":
            out["limit_price"] = trade.limit_price           # the resting order's price: a number code made
        return out

    def notify_account_manager(self) -> dict:
        """Close the run: ask the Account Manager for the account as it stands now. Every fill was already reported to the Account Manager the moment it was filled (place_order told you the cash left), so this tells it nothing new. Call it once, after every trade is placed; it is refused while any trade is still "new" and names it.

        A second call gives the first answer again. After it nothing more can be placed. It returns the cash before the run and after it (the Account Manager's own figures), how many trades were filled, rejected and are still active, and whether the fills were booked in the account book.
        """
        waiting = self.broker.tickets()
        if waiting and not getattr(self.notifier, "done", False):
            raise ToolInputError(f"{waiting[0].trade_id} is still new: place every trade before you notify the Account Manager")
        return self.notifier()

    # ---------------------------------------------------------------- for LangChain

    def langchain_tools(self) -> list[BaseTool]:
        """The allow-listed tools as LangChain tools, in the order of TRADER_TOOLS."""
        return [tool(getattr(self, name), parse_docstring=True) for name in TRADER_TOOLS]


def tool_content(output) -> str:
    """What a tool result looks like to the model: compact JSON."""
    return dumps(output, ensure_ascii=False, separators=(",", ":"))      # the one encoder: a Decimal is a number
