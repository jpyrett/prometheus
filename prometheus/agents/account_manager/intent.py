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
"""
import json
import math
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from prometheus.rag.loaders import check_symbol

MAX_TEXT = 600
MAX_COUNT = 10
MAX_BUDGET = 1_000_000_000
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 &.,'’/+-]{0,59}$")
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
BUY_CUE = re.compile(r"\b(?:buy|buying|bought|purchase|purchasing|invest|investing|acquire|get|trade|trading|spend|spending|own|add|(?:put|place)\b[^.]{0,40}\binto|pick up|load up|go long)\b", re.I)
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


class TradeIntent(BaseModel):
    """The user's request as the Account Manager understood it. `request` is what they typed (kept for the record and the report)."""
    model_config = ConfigDict(extra="forbid")

    request: str
    action: Literal["buy", "research"]
    topic: str | None = None
    symbols: list[str] = Field(default_factory=list)
    companies: list[str] = Field(default_factory=list)
    count: int | None = Field(default=None, ge=1, le=MAX_COUNT)
    allocation: Literal["even", "default"] = "default"      # "even": the user asked to split the money equally; "default": no split was asked (the Account Manager's own way for the mode: round-robin in Auto, what the limits allow in Ask; it was called "advisor" before entry 80, when the Advisor sized)
    budget: float | None = Field(default=None, gt=0, le=MAX_BUDGET)

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
        return self.budget is not None or bool(HOW_MUCH.search(self.request))

    def what(self) -> str:
        """What is to be bought or looked at, in words: the theme, or the names, or the tickers."""
        if self.topic:
            return f"{self.topic!r} companies"
        return ", ".join([*self.symbols, *self.companies])

    def describe(self) -> str:
        """One plain sentence of what was understood, for the user to confirm before anything runs."""
        verb = "buy" if self.action == "buy" else "look at (nothing will be bought)"
        count = f"the {self.count} most recommended " if self.count else "the most recommended "
        spread = "; the money spread evenly over the companies that are approved" if self.allocation == "even" else ""
        spend = f"; spending at most ${self.budget:,.2f}" if self.budget else "; spending as much as the account and its risk limits allow" if self.action == "buy" else ""
        return f"{verb} {count}{self.what()}{spread}{spend}"

    def as_request(self) -> tuple[str, str]:
        """(text, kind) for `recommendations.classify`: a theme is a phrase, tickers are symbols, a name is a company, several names a list. Raises ValueError for a mix the runner has no single way to ask."""
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


numbers_in = _numbers      # the public name: `clarify.py` checks the amount of an answer against the numbers the user typed


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 3}


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
    known = {"action", "topic", "symbols", "companies", "count", "allocation", "budget"}
    if set(raw) - known:
        raise ValueError(f"unknown fields: {', '.join(sorted(map(str, set(raw) - known)))}")
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
    return intent
