"""What the user asked for, as a checked form (design choices entry 78).

    intent = parse_intent(reply_text, typed)          # ValueError, with the reason, when the model's reply is not a form that the user's own words support
    intent.describe()                                 # "buy the 3 most recommended companies of 'oil', spending ..."
    text, kind = intent.as_request()                  # for `recommendations.classify(text, kind)`

THE MODEL MAPS WORDS; CODE CHECKS THEM. The Account Manager's model (`AccountManager.understand_request`) fills in the form; this module accepts it only when the user's own text supports every field:
  * a ticker is in the text as typed, in capitals, and a company or brand name is in the text (case ignored);
  * the topic shares a word with the text;
  * the count is a number or a number word (one to ten) that is in the text, the budget a dollar amount that is in the text ($5,000, 5000, 5k);
  * "even" needs a cue word (even, evenly, equal, balance, split, divide, same amount), and "buy" a buying word (buy, purchase, invest, acquire, get, trade, spend, own, put ... into). "default" (no split
    asked) needs nothing: it is what every other request means.
A field the text does not support makes the whole form refused, with the reason, so that the model is asked once more and the user is never shown a company they did not name. Nothing here is a sector, a
company or a ticker: any industry, any list of tickers and any company name goes through the same checks. No network and no model in this module.

A DIRECT ORDER (design choices entries 87 and 88). When the user names the trade itself ("Buy 50 shares of MSFT", "sell 10 shares of Tesla") nothing is to be recommended or reviewed: the form is `action: "order"`
with `orders`, one {side, symbol or company, shares} per trade, and the desk takes the order straight to the Account Manager and the Trader. The checks are the same kind: the ticker, or the company's name, is in the
text as typed, the number of shares is written as a number of SHARES right before it ("50 shares of MSFT", "50 MSFT", "50 shares of Microsoft Corp (MSFT)", "50 shares of Microsoft": a dollar amount or a percentage is not
a number of shares), and the last buying or selling word before it is the order's side. A COMPANY's name is not turned into a ticker here: the Account Manager asks the Data Clerk for the symbol (`DataClerk.request_symbols`),
and in Ask mode the user confirms what was found. A form that holds anything but the orders (a theme, a count, a budget) is refused.
"""
import json
import math
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from prometheus.common.work_order import DIRECT_SIDES, MAX_DIRECT_SHARES
from prometheus.rag.loaders import check_symbol

MAX_TEXT = 600
MAX_COUNT = 10
MAX_BUDGET = 1_000_000_000
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 &.,'’/+-]{0,59}$")
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
BUY_CUE = re.compile(r"\b(?:buy|buying|bought|purchase|purchasing|invest|investing|acquire|get|trade|trading|spend|spending|own|add|(?:put|place)\b[^.]{0,40}\binto|pick up|load up|go long)\b", re.I)
PLAIN_TICKER = re.compile(r"\$?[A-Z0-9][A-Z0-9.\-]{0,9}")      # a ticker in capitals (the same shape as `symbols.is_plain_ticker`)
SELL_CUE = re.compile(r"\b(?:sell|selling|sold|unload|dump|liquidate|trim|exit|get rid of)\b", re.I)
EVEN_CUE = re.compile(r"\b(?:even|evenly|equal|equally|balanc\w*|split|divid\w*|same (?:amount|dollar\w*|money))\b", re.I)
MONEY = re.compile(r"\$?\s?(\d[\d,]*(?:\.\d+)?)\s*(k\b|thousand\b|m\b|million\b)?", re.I)
# What the user says when they have already said HOW MUCH to buy (design choices entry 85): a percentage, a dollar amount, a number of shares, or "as much as I can". The Account Manager asks how much only when none of
# these is in the request. It is a program's check of the user's own words, never a model's.
HOW_MUCH = re.compile(r"(?:\d[\d,]*(?:\.\d+)?\s?(?:%|percent\b|per cent\b)"                                          # 20%, 20 percent
                      r"|\$\s?\d"                                                                                   # $3,000
                      r"|\d[\d,]*(?:\.\d+)?\s*(?:k|thousand|million)?\s*(?:dollars?|bucks|usd)\b"                      # 3000 dollars
                      r"|\b(?:\d[\d,]*|one|two|three|four|five|six|seven|eight|nine|ten)\s+shares?\b"                  # 10 shares
                      r"|\bas (?:much|many)\b|\ball[- ]in\b|\beverything\b|\bfully invest\w*"                             # as much as I can
                      r"|\ball (?:of )?(?:my|the|our)\b[^.]{0,20}\b(?:money|cash|funds|account)"                          # all of my money
                      r"|\b(?:the )?(?:most|max(?:imum)?)\s+(?:amount|shares?|stocks?|money)\b)", re.I)                  # the most amount, the maximum shares


class DirectOrder(BaseModel):
    """One trade the user named: `buy` or `sell`, a whole number of shares (None when they wrote none: "I want to buy MSFT"; the Account Manager then asks them in Ask mode, or works it out, design choices entry 98), and the stock
    either as a ticker (`symbol`) or as a company's name (`company`; the Data Clerk finds its symbol: entry 88). Exactly one of the two."""
    model_config = ConfigDict(extra="forbid")

    side: Literal["buy", "sell"]
    symbol: str | None = None
    company: str | None = None
    shares: int | None = Field(default=None, ge=1, le=MAX_DIRECT_SHARES)

    @field_validator("symbol")
    @classmethod
    def _symbol(cls, value):
        return None if value is None else check_symbol(value.strip().lstrip("$"))

    @field_validator("company")
    @classmethod
    def _company(cls, value):
        if value is None:
            return None
        value = " ".join(value.split())
        if not NAME.match(value):
            raise ValueError("a company name is not a few plain words")
        if PLAIN_TICKER.fullmatch(value):
            raise ValueError(f"{value!r} is a ticker: a ticker goes in symbol, and a company's name in company")
        return value

    @model_validator(mode="after")
    def _one_way_to_name_the_stock(self):
        if (self.symbol is None) == (self.company is None):
            raise ValueError("an order names the stock by its ticker (symbol) or by its company name (company), one of them")
        return self

    @property
    def named(self) -> str:
        """What the user called the stock: the ticker, or the company's name as typed."""
        return self.symbol or self.company

    def describe(self) -> str:
        if self.shares is None:
            return f"{self.side} {self.named} (you did not say how many shares)"
        return f"{self.side} {self.shares:,} share{'s' if self.shares != 1 else ''} of {self.named}"


class TradeIntent(BaseModel):
    """The user's request as the Account Manager understood it. `request` is what they typed (kept for the record and the report)."""
    model_config = ConfigDict(extra="forbid")

    request: str
    action: Literal["buy", "research", "order"]      # "order": the user named the trades themselves (`orders`): a direct order, entry 87
    topic: str | None = None
    symbols: list[str] = Field(default_factory=list)
    companies: list[str] = Field(default_factory=list)
    count: int | None = Field(default=None, ge=1, le=MAX_COUNT)
    allocation: Literal["even", "default"] = "default"      # "even": the user asked to split the money equally; "default": no split was asked (the Account Manager's own way for the mode: round-robin in Auto, what the limits allow in Ask; it was called "advisor" before entry 80, when the Advisor sized)
    budget: float | None = Field(default=None, gt=0, le=MAX_BUDGET)
    orders: list[DirectOrder] = Field(default_factory=list, max_length=MAX_COUNT)      # only with action "order"

    @model_validator(mode="after")
    def _orders_go_with_the_order_action(self):
        if (self.action == "order") != bool(self.orders):
            raise ValueError("orders: a direct order has the action 'order' and at least one trade, and no other action has any")
        if self.action == "order":
            numbered = any(o.shares is not None for o in self.orders)
            if self.topic or self.symbols or self.companies or self.count is not None or ((self.budget is not None or self.allocation != "default") and numbered):
                raise ValueError("orders: a direct order names the trades and nothing else (dollars to spend and an even split only go with a stock named without a number of shares): ask for recommendations in a separate request")
            if len({o.named.casefold() for o in self.orders}) != len(self.orders):
                raise ValueError("orders: one trade for each stock")
        return self

    @field_validator("topic")
    @classmethod
    def _topic(cls, value):
        if value is None:
            return None
        value = " ".join(value.split())
        if not value:
            return None
        if not NAME.match(value):
            raise ValueError("the topic is not a few plain words")
        return value

    @field_validator("companies")
    @classmethod
    def _companies(cls, values):
        clean = [" ".join(v.split()) for v in values]
        if any(not NAME.match(v) for v in clean):
            raise ValueError("a company name is not a few plain words")
        return list(dict.fromkeys(clean))

    @field_validator("symbols")
    @classmethod
    def _symbols(cls, values):
        return list(dict.fromkeys(check_symbol(v.strip().lstrip("$")) for v in values))

    def said_how_much(self) -> bool:
        """Whether the user's own words already say how much to buy: a dollar amount (the budget), a percentage, a number of shares, or "as much as I can" and its kin. When they do not, the Account Manager asks in Ask
        mode (`clarify.py`). A check of the typed words by a program, with no model."""
        if self.action == "order":
            return all(o.shares is not None for o in self.orders) or self.budget is not None or bool(AS_MUCH.search(self.request))      # an order with no number is asked about unless the user said dollars or "as much as I can"
        return self.budget is not None or bool(HOW_MUCH.search(self.request))

    def what(self) -> str:
        """What is to be bought or looked at, in words: the theme, or the names, or the tickers."""
        if self.action == "order":
            return ", ".join(o.named for o in self.orders)
        if self.topic:
            return f"{self.topic!r} companies"
        return ", ".join([*self.symbols, *self.companies])

    def describe(self) -> str:
        """One plain sentence of what was understood, for the user to confirm before anything runs."""
        if self.action == "order":
            spend = f", spending ${self.budget:,.2f}" + (" spread evenly" if self.allocation == "even" else "") if self.budget else ""
            return ", ".join(o.describe() for o in self.orders) + spend + " (your own order: no recommendation is made and the Critic is not asked)"
        verb = "buy" if self.action == "buy" else "look at (nothing will be bought)"
        count = f"the {self.count} most recommended " if self.count else "the most recommended "
        spread = "; the money spread evenly over the companies that are approved" if self.allocation == "even" else ""
        spend = f"; spending at most ${self.budget:,.2f}" if self.budget else "; spending as much as the account allows" if self.action == "buy" else ""
        return f"{verb} {count}{self.what()}{spread}{spend}"

    def as_request(self) -> tuple[str, str]:
        """(text, kind) for `recommendations.classify`: a theme is a phrase, tickers are symbols, a name is a company, several names a list. Raises ValueError for a mix the runner has no single way to ask."""
        if self.action == "order":
            raise ValueError("a direct order is not a search for companies: it goes straight to the Account Manager")
        named = [*self.symbols, *self.companies]
        if self.topic and named:
            raise ValueError("you gave both a theme and named companies: ask for one of them at a time")
        if self.topic:
            return self.topic, "phrase"
        if self.symbols and not self.companies:
            return " ".join(self.symbols), "symbols"
        if len(named) == 1:
            return named[0], "company"
        return ", ".join(named), "list"


def _numbers(text: str) -> set[float]:
    """Every number in the text: digits ($5,000, 5k, 2.5 million) and number words (three)."""
    found = set()
    for match in MONEY.finditer(text):
        value = float(match.group(1).replace(",", ""))
        unit = (match.group(2) or "").lower()
        found.add(value * (1000 if unit in ("k", "thousand") else 1_000_000 if unit in ("m", "million") else 1))
    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", text, re.I):
            found.add(float(value))
    return found


AS_MUCH = re.compile(r"\bas (?:much|many)\b|\ball[- ]in\b|\beverything\b|\bfully invest\w*|\ball (?:of )?(?:my|the|our)\b[^.]{0,20}\b(?:money|cash|funds|account)", re.I)      # "as much as I can" and its kin
# What makes a request one for RECOMMENDATIONS and not a direct order (entry 98: a named stock is the user's own trade unless they ask for a recommendation): checked by a program in the user's own words.
RECOMMEND_CUE = re.compile(r"\b(?:recommend\w*|advi[cs]e\w*|research\w*|analy[sz]\w*|should i|worth (?:buying|it)|if (?:the |it |they |that )?(?:pass\w*|approv\w*|looks? good|is good|are good))\b", re.I)


numbers_in = _numbers      # the public name: `clarify.py` checks the amount of an answer against the numbers the user typed


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 3}


def _cue_before(typed: str, position: int) -> str | None:
    """The side of the LAST buying or selling word before `position` in the user's words ("buy 50 MSFT, sell 10 TSLA": "buy" before the 50, "sell" before the 10), or None when there is none."""
    cues = [(m.start(), "buy") for m in BUY_CUE.finditer(typed, 0, position)] + [(m.start(), "sell") for m in SELL_CUE.finditer(typed, 0, position)]
    return max(cues)[1] if cues else None


def _order_phrases(typed: str, name: str, ticker: bool = True) -> list[tuple[int, int]]:
    """Every (number of shares, where it starts) the user wrote right before this ticker or company name: "50 shares of MSFT", "50 shares in MSFT", "50 MSFT", number words up to ten, and up to three words in between
    ("50 shares of Microsoft Corp (MSFT)"). A dollar amount, a percentage or a decimal is not a number of shares and is never read as one, and neither is the last word of a hyphenated number ("twenty-five" is not "five")."""
    words = "|".join(NUMBER_WORDS)
    stock = re.escape(name).replace(r"\ ", r"\s+")
    pattern = re.compile(rf"(?<![\w$.,%-])(\d[\d,]*|{words})(?![\d.,%])\s+(?:shares?\s+)?(?:(?:of|in)\s+)?(?:[A-Za-z&'’-]+\s+){{0,3}}\(?\$?{stock}(?![A-Za-z0-9])", re.I)
    found = []
    for match in pattern.finditer(typed):
        word = match.group(1).lower()
        found.append((NUMBER_WORDS[word] if word in NUMBER_WORDS else int(word.replace(",", "")), match.start()))
    return found


def check_order(order: DirectOrder, typed: str) -> None:
    """Raises ValueError, with the reason, unless the user's own words give this trade: the ticker as typed (or the company's name, case ignored), the shares as a number of shares right before it, and the side of the
    buying or selling word before that."""
    if order.symbol is not None:
        if not re.search(rf"(?<![A-Za-z0-9]){re.escape(order.symbol)}(?![A-Za-z0-9])", typed):
            raise ValueError(f"the ticker {order.symbol} is not in what you typed")
    elif not re.search(r"\s+".join(map(re.escape, order.company.split())), typed, re.I):
        raise ValueError(f"the company {order.company!r} is not in what you typed")
    if RECOMMEND_CUE.search(typed):
        raise ValueError("you asked for recommendations or advice (or made the trade depend on them): that is not a direct order, so it is a request to buy or research")
    if order.shares is None:
        typed_numbers = _order_phrases(typed, order.named, ticker=order.symbol is not None)
        if typed_numbers:
            raise ValueError(f"you typed {typed_numbers[0][0]:,} shares of {order.named}: put the number in shares, not null")
        position = next((m.start() for m in re.finditer(r"(?<![A-Za-z0-9])" + r"\s+".join(map(re.escape, order.named.split())) + r"(?![A-Za-z0-9])", typed, 0 if order.symbol else re.I)), None)
        if position is None or _cue_before(typed, position) != order.side:
            word = "buying word (buy, purchase, get ...)" if order.side == "buy" else "selling word (sell, unload, dump ...)"
            raise ValueError(f"{order.side} {order.named} is not what you typed: the last word about buying or selling before {order.named} is not a {word}")
        return
    phrases = [at for shares, at in _order_phrases(typed, order.named, ticker=order.symbol is not None) if shares == order.shares]
    if not phrases:
        raise ValueError(f"{order.shares:,} shares of {order.named} is not what you typed: say it as a number of shares, for example 'buy {order.shares:,} shares of {order.named}'")
    if not any(_cue_before(typed, at) == order.side for at in phrases):
        word = "buying word (buy, purchase, get ...)" if order.side == "buy" else "selling word (sell, unload, dump ...)"
        raise ValueError(f"{order.side} {order.named} is not what you typed: the last word about buying or selling before the {order.shares:,} shares is not a {word}")


def check_typed(typed) -> str:
    """The user's words with the spaces tidied. Raises ValueError for nothing at all or for more than the form can read: a request refused here costs no model call."""
    typed = " ".join(typed.split()) if isinstance(typed, str) else ""
    if not typed:
        raise ValueError("there is nothing to understand: type what you want to buy or look at")
    if len(typed) > MAX_TEXT:
        raise ValueError(f"the request is longer than {MAX_TEXT} characters")
    return typed


def parse_intent(reply: str, typed: str) -> TradeIntent:
    """The form in the model's reply, accepted only when `typed` (the user's own words) supports every field. Raises ValueError with a short reason for the model (and the user) otherwise."""
    typed = check_typed(typed)
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", (reply or "").strip())
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found")
    try:
        raw = json.loads(body[start:end + 1])
    except json.JSONDecodeError:
        raise ValueError("not valid JSON") from None
    if not isinstance(raw, dict):
        raise ValueError("not a JSON object")
    known = {"action", "topic", "symbols", "companies", "count", "allocation", "budget", "orders"}
    if set(raw) - known:
        raise ValueError(f"unknown fields: {', '.join(sorted(map(str, set(raw) - known)))}")
    orders = raw.get("orders", [])
    if not isinstance(orders, list) or not all(isinstance(o, dict) and {"side"} <= set(o) <= {"side", "symbol", "company", "shares"} and ("symbol" in o) != ("company" in o)
                                                and all(isinstance(o[k], str) for k in ("symbol", "company") if k in o) and (o.get("shares") is None or (isinstance(o["shares"], int) and not isinstance(o["shares"], bool))) for o in orders):
        raise ValueError('orders must be a list of {"side": "buy" or "sell", "symbol": a ticker in capitals, or "company": a company name, "shares": a whole number, or null when the person wrote no number of shares}')
    for number, o in enumerate(orders):
        if "symbol" in o:
            try:
                clean = check_symbol(o["symbol"].strip().lstrip("$"))
            except ValueError:
                clean = None
            if clean != o["symbol"].strip().lstrip("$"):
                raise ValueError(f"orders[{number}]: {o['symbol']!r} is not a ticker in capitals: a company's name goes in company, not in symbol")
    if raw.get("action") == "order" and RECOMMEND_CUE.search(typed):      # the words ask for a recommendation (entry 98): not a direct order, whatever the model said; the stocks it listed are the ones to be recommended
        raw = {**raw, "action": "buy", "orders": [], "symbols": [o["symbol"].strip().lstrip("$") for o in orders if "symbol" in o], "companies": [" ".join(o["company"].split()) for o in orders if "company" in o]}
        orders = []
    if raw.get("action") == "order":     # the model repeated the tickers or the names of the orders in `symbols` or `companies` as well: they are the orders' own, so nothing is lost
        own_symbols = {o["symbol"].strip().lstrip("$") for o in orders if "symbol" in o}
        own_companies = {" ".join(o["company"].split()).casefold() for o in orders if "company" in o}
        if isinstance(raw.get("symbols"), list) and all(isinstance(v, str) for v in raw["symbols"]) and {v.strip().lstrip("$") for v in raw["symbols"]} <= own_symbols:
            raw = {**raw, "symbols": []}
        if isinstance(raw.get("companies"), list) and all(isinstance(v, str) for v in raw["companies"]) and {" ".join(v.split()).casefold() for v in raw["companies"]} <= own_companies:
            raw = {**raw, "companies": []}
    for key in ("symbols", "companies"):
        if not isinstance(raw.get(key, []), list) or not all(isinstance(v, str) for v in raw.get(key, [])):
            raise ValueError(f"{key} must be a list of text")
    if isinstance(raw.get("count"), bool) or isinstance(raw.get("budget"), bool):
        raise ValueError("count and budget must be numbers")
    try:
        intent = TradeIntent(request=typed, **raw)
    except ValidationError as error:
        first = error.errors()[0]
        raise ValueError(f"{'.'.join(str(x) for x in first['loc']) or 'the form'}: {first['msg']}") from None
    except ValueError as error:   # check_symbol
        raise ValueError(str(error)) from None
    # every field must be in the user's own words
    lowered, words, numbers = typed.lower(), _words(typed), _numbers(typed)
    for order in intent.orders:
        check_order(order, typed)
    if intent.action == "order":
        return intent
    if not (intent.topic or intent.symbols or intent.companies):
        raise ValueError("no theme, ticker or company: name what you want to buy or look at")
    for symbol in intent.symbols:
        if not re.search(rf"(?<![A-Za-z0-9]){re.escape(symbol)}(?![A-Za-z0-9])", typed):
            raise ValueError(f"the ticker {symbol} is not in what you typed")
    for name in intent.companies:
        if name.lower() not in lowered:
            raise ValueError(f"the company {name!r} is not in what you typed")
    if intent.topic and not (_words(intent.topic) & words):
        raise ValueError(f"the theme {intent.topic!r} shares no word with what you typed")
    if intent.count is not None and float(intent.count) not in numbers:
        raise ValueError(f"the number {intent.count} is not in what you typed")
    if intent.budget is not None and not any(math.isclose(intent.budget, n, rel_tol=1e-9, abs_tol=0.005) for n in numbers):
        raise ValueError(f"the amount {intent.budget:g} is not in what you typed")
    if intent.allocation == "even" and not EVEN_CUE.search(typed):
        raise ValueError("an even split was not asked for in what you typed")
    if intent.action == "buy" and not BUY_CUE.search(typed):
        raise ValueError("buying was not asked for in what you typed")
    return as_direct_order(intent, typed) or intent


def as_direct_order(intent: TradeIntent, typed: str) -> TradeIntent | None:
    """A request to BUY stocks the user NAMED, with nothing that asks for a recommendation, is a direct order (design choices entry 98: a recommendation is not needed for a direct order unless the user asks for one).
    The model sometimes reads such a request as "buy" (a list of tickers and the choice left to the desk) and sometimes as "order"; this is the program's check of the user's own words, so the answer does not depend on
    the model: named tickers or companies, no theme, no count ("the best 2 of ..."), no word that asks for a recommendation, and the user's own buying word before each stock. Returns the direct order, or None when the
    request is not one (it stays what the model said)."""
    if intent.action != "buy" or intent.topic or intent.count is not None or not (intent.symbols or intent.companies) or RECOMMEND_CUE.search(typed):
        return None
    try:
        orders = [DirectOrder(side="buy", symbol=symbol) for symbol in intent.symbols] + [DirectOrder(side="buy", company=name) for name in intent.companies]
        converted = TradeIntent(request=intent.request, action="order", orders=orders, budget=intent.budget, allocation=intent.allocation)
        for order in converted.orders:
            check_order(order, typed)
    except ValueError:      # includes a pydantic ValidationError
        return None
    return converted
