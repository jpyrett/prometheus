"""What the user asked for, as a checked form (design choices entry 78).

    intent = parse_intent(reply_text, typed)          # ValueError, with the reason, when the model's reply is not a form that the user's own words support
    intent.describe()                                 # "buy the 3 most recommended companies of 'oil', spending ..."
    text, kind = intent.as_request()                  # for `recommendations.classify(text, kind)`

THE MODEL MAPS WORDS; CODE CHECKS THEM. The Account Manager's model (`AccountManager.understand_request`) fills in the form; this module accepts it only when the user's own text supports every field:
  * a ticker is in the text as typed, in capitals, and a company or brand name is in the text (case ignored);
  * the topic shares a word of three letters or more with the text, or, when every word of the topic is shorter (AI, EV, 5G, IT), every word of it is a whole word of the text (entry 113);
  * the count is a number or a number word (one to ten) that is in the text, the budget a dollar amount that is in the text ($5,000, 5000, 5k);
  * "even" needs a cue word (even, evenly, equal, balance, split, divide, same amount), and "buy" a buying word (buy, purchase, invest, acquire, get, trade, spend, own, put ... into). "default" (no split
    asked) needs nothing: it is what every other request means.
A field the text does not support makes the whole form refused, with the reason, so that the model is asked once more and the user is never shown a company they did not name. Nothing here is a sector, a
company or a ticker: any industry, any list of tickers and any company name goes through the same checks. No network and no model in this module.

THE THEME (design choices entry 113). The prompt tells the model to leave the word "stocks" out of the topic, so "the best AI stocks" is the topic `AI`, a word of two letters: the check "shares a word of three letters or
more" could never pass for a theme made only of short words, so such a theme is checked by ALL its words instead (each one a whole word of what was typed; a theme whose words are not in the text is refused as before).
And a phrase the user put IN QUOTES is their theme, word for word: when the request is a search (buy or research, not an order) with a topic and no ticker and no company, and every word of the model's topic is a word of
ONE quoted phrase, the theme becomes that phrase as typed, spaces collapsed ("AI stocks"), and that is what the Data Clerk is asked for (`TradeIntent.as_request`). A PROGRAM decides it from the user's words (`quoted_theme`);
the phrase is used only if it passes the topic's own check (`NAME`) and the Data Clerk's phrase rule (`CLERK_PHRASE`), otherwise the model's topic stays. A direct order, a ticker and a company's name are not touched.

A DIRECT ORDER (design choices entries 87 and 88). When the user names the trade itself ("Buy 50 shares of MSFT", "sell 10 shares of Tesla") nothing is to be recommended or reviewed: the form is `action: "order"`
with `orders`, one {side, symbol or company, shares} per trade, and the desk takes the order straight to the Account Manager and the Trader. The checks are the same kind: the ticker, or the company's name, is in the
text as typed, the number of shares is written as a number of SHARES right before it ("50 shares of MSFT", "50 MSFT", "50 shares of Microsoft Corp (MSFT)", "50 shares of Microsoft": a dollar amount or a percentage is not
a number of shares), and the last buying or selling word before it is the order's side. THE NUMBER OF SHARES MAY HAVE A DECIMAL PART (design choices entry 110): "sell 0.5 shares of MSFT", "sell 81.004455 shares of DPZ"; at most
six decimals (`shares.PLACES`), and a number with more is refused with that reason, never rounded; without it a holding of less than one share, or the fraction left after a sale, could never be sold. A COMPANY's name is not turned into a ticker here: the Account Manager asks the Data Clerk for the symbol (`DataClerk.request_symbols`),
and in Ask mode the user confirms what was found. A form that holds anything but the orders (a theme, a count, a budget) is refused.

AN AMOUNT FOR EACH STOCK (design choices entry 122; design/direct_dollars_each_spec.md). A direct order has the number of SHARES the user wrote for it (`DirectOrder.shares`), or the DOLLARS they wrote for that stock (`DirectOrder.dollars`:
"Buy $3,000 of XOM and $3,000 of MSFT" is two orders of $3,000), or neither, never both; ONE total for several stocks is the `budget` ("Buy $5,000 of MU and AMD"), and a request holds a budget or the dollars of its orders, not both.
The model reads it; CODE checks the number (`check_dollars`): the dollars of an order must be an amount the user WROTE AS MONEY (`money_written`: `$3,000`, `3,000 dollars`, `$3k`; a bare number, `3k` alone, a number of shares and a
percentage are not), positive and to the cent (`Money`: a bool and a string are refused), and an amount is the dollars of as many orders as it was WRITTEN for (`check_dollars_each`: "Buy $5,000 of MU and AMD"
writes $5,000 once, so it is one total, never $5,000 for each: a number counted against a number). No word before or after the stock is looked at. And a request that wrote an amount of money with a `$` or the word dollars
(`money_written(strict=True)`) but whose reading has no budget, no dollars and no shares on any order is refused (`check_amount_was_read`): the Account Manager never falls back to spending all the cash on a request that said an amount
(the model is asked once more, and then the user is asked to clarify).

"EACH" (design choices entry 125; design/dollars_each_word_spec.md). ONE amount the user wrote once may be the amount of EVERY stock, but only when they said so: `TradeIntent.each` is true when the model read that the amount applies to every stock
individually ("Buy $3,000 each of XOM and MSFT", "$2,000 apiece", "$1,000 per stock"); with it the amount is the dollars (or the shares) of EVERY order, the same in each, and for the recommendation path ("the top 3 with $3,000 each") it is the
`budget`, which then means the amount for EACH purchase, never a total (`TradeIntent.each_dollars`; `total_budget` is the one total). CODE checks it (`check_each`): a word that says each is in the user's text (`EACH_CUE`, a small closed list of whole words: it decides
how much money is spent, so a program reads it, not the model), every order carries the same amount, and the amount was written as money (or, for shares, as a number of shares). `each` with no such word is a reading error, and without `each` the earlier rule
stands: an amount written once for several stocks is ONE total (`check_dollars_each`).

DECIMAL MONEY (design choices entry 112): `TradeIntent.budget` is a `Money` (a Decimal to the cent: the dollars the user typed, read from the DIGITS the model's JSON wrote, a bool and a string refused) and `DirectOrder.shares` a share quantity
(a Decimal); the numbers of the user's own words (`numbers_in`) are Decimals read from their digits ("$5,000", "5k", "2.5 million", "three"), and the checker compares the form with them exactly: the amount is in the words or the form is refused.

DOLLARS OR SHARES, A PRICE LIMIT, "SKIP RECOMMENDATIONS" (design choices entry 138; design/request_numbers_spec.md; Jacob, 2026-11-20: *"if they use a number with a $ (dollar sign) or word bucks or dollars then they are referring to using
price. Without these they are referring to quantity"*). A PROGRAM decides what a number is from how it was WRITTEN; the model's prompt says the same, and code has the last word:
  * MONEY is a number with a `$` before it or the word dollars (bucks, usd) after it ("$1000", "1000 bucks", "$5k", "5 thousand dollars"); `3k` or `5 thousand` alone is not money (`money_amounts`). A percentage stays a percentage.
  * SHARES are a bare number right before a stock, with or without `shares`, `of`, `in` ("10 of MU", "1000 MU", "10 shares of MU"): `_order_phrases`, which no longer reads "500 bucks of AMD" or "1000 dollars of MU" as shares.
  * A number the model put in the wrong kind is CORRECTED here, never guessed the other way: dollars that were written as a bare number before the stock become shares, shares that were written as money become dollars, a number the model read
    as the budget or the count of "buy 10 of MU" becomes the shares of that stock, and the correction is on the intent (`TradeIntent.corrected`, shown by `describe()`).
  * A bare number after a money verb ("spend 1000 on MU", "invest 1000 in MU", "put 1000 into MU") is AMBIGUOUS (`ambiguous_numbers`): the number is taken out of the reading and kept as `TradeIntent.unclear_number`; the desk asks the user in Ask
    mode ("1000 shares or $1,000.00?": `AccountManager.settle_unclear_number`, which reads the answer with `read_unit_answer`) and stops in Auto mode. It is never bought as 1000 shares without a question.
  * A PRICE LIMIT is a `$` amount tied to ONE share ("under $50 a share", "$50 per share", "at most $50 a share") or after `under`, `below`, `less than`, `cheaper than`, `priced under` when it is not followed by `of`, `worth`, `in` a stock ("stocks
    below $20"): `TradeIntent.max_price` (a `Price`), copied to every `DirectOrder`; it is never a money amount (`written_amounts` takes its span out of `money_amounts`). `at most $5,000` and `no more than $5,000` with no share beside them are
    amounts to spend. (A FLOOR, "over $20 a share", was ignored with a note until entry 140; see below.)
  * A PRICE FLOOR (entry 140; design/limit_orders_spec.md, F1) is the same from below: a `$` amount tied to one share after `over`, `above`, `more than`, `at least`, `no less than`, or followed by `or more`, `or higher` ("over $20 a share",
    "sell 10 shares of MSFT at $600 or more"): `TradeIntent.min_price`, copied to every `DirectOrder`. `or less` and `or lower` after an amount are a ceiling, `or better` is a floor for a SALE and a ceiling for a purchase (the side is the last
    buying or selling word before it), and "between $20 and $60 a share" is both. An amount with no share beside it counts as a price only after `at`, `price` or `limit`; "with $1,000 or more" and "invest $5,000 or less" are money. A floor
    above the ceiling, two floors, or a floor with both a purchase and a sale in one request are `NeedsClarification` (a purchase that waits for the price to RISE is a stop order: not built). It is never an amount to spend.
  * A number of SHARES TYPED WITH A RECOMMENDATION (entry 140, C1) is kept: "buy 10 of MU if the recommendation passes" is `TradeIntent.shares_of` {"MU": 10}, "buy 5 shares of each of the top 3 oil stocks if recommended" is
    `TradeIntent.shares_each` 5. A PROGRAM reads them from the words by the same code as a direct order's shares (`typed_shares`, `_order_phrases`, `SHARES_EACH`; the count of a request is never shares, a fraction of a share is refused: a
    recommendation is bought in whole shares), the model's budget or `each` that took the number for money is corrected (`TradeIntent.corrected`), and `said_how_much()` is true when every stock named has its number or `shares_each` is set.
    The Account Manager makes them each approved purchase's limit (`decisions.typed_wanted`).
  * PERMISSION TO RAISE FUNDS (entry 142; design/direct_raise_funds_spec.md, R2): "Buy 2 shares of ANT and raise funds if needed" says the Account Manager may SELL some of the user's holdings to pay for the purchase when the cash does not
    cover it. A PROGRAM decides it from the words (`RAISE_FUNDS_CUE`: raise funds / cash / money, free up cash, "sell some stock to pay for it", "take profits to buy"), and a negation wins ("don't raise funds", "do not sell anything",
    `NO_RAISE_FUNDS`): `TradeIntent.raise_funds`. The model's form has the field too and only has to agree: a `true` the words do not support is refused, a `false` is overruled by the words. Without the words (or the setting
    ALWAYS_RAISE_FUNDS, entry 142 R12) nothing is sold: Ask mode asks once, Auto mode says what to add.
  * THE NEWS (entry 149; design/news_view_spec.md; Jacob, 2026-11-23: *"Show the news for Tesla"*, *"Whats Tesla doing the last week?"*): a request that ONLY asks to see the news of one or more stocks is `action: "news"`. THE MODEL reads it (no cue-word
    filter decides it; a request that also asks to buy, sell, recommend or research is not news). It holds `symbols` and/or `companies` (at most `news.max_stocks`) or a `topic` searched as a term, and the window `news_days`, and NOTHING that moves money
    or asks for a trade: the validator refuses orders, a budget, `each`, a count, an even split, `raise_funds`, a price limit or floor and typed shares, each with its reason. A PROGRAM reads an explicit window from the words and wins over the model
    (`news_window`: "last 3 days", "past 2 weeks", "last week", "this month", "today", "yesterday"); with none written the model's value, and with none either `news.default_days`; a window over `news.max_days` (90) is cut and said (`corrected`).
  * "Skip recommendations" is not a request for them: a recommend cue inside a NEGATED span ("skip recommendations", "no research needed", "without the advisor") is no cue (`NO_RECOMMEND`, `asks_for_recommendation`), so "Buy $2500 worth of
    symbol KO and skip recommendations." is a direct order of $2,500 of KO. A request that skips them and names no stock to buy is `NeedsClarification` ("name the stocks to buy, or let me recommend some").
"""
import json
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StrictBool, ValidationError, field_validator, model_serializer, model_validator

from prometheus.agents.account_manager.config import load_config
from prometheus.common import shares as sh
from prometheus.common.money import Digits, PositiveMoney, Price, canonical, digits
from prometheus.common.work_order import DIRECT_SIDES, MAX_DIRECT_SHARES
from prometheus.rag.loaders import check_symbol

MAX_TEXT = 600
MAX_COUNT = 10
MAX_NEWS_DAYS = 90      # the news fetch's own limit (`work_order.MAX_DAYS["news"]`, `news.max_days` of the Account Manager's config): a window above it is cut (entry 149)
MAX_BUDGET = Decimal(1_000_000_000)
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 &.,'’/+-]{0,59}$")
CLERK_PHRASE_CHARS = r"[A-Za-z0-9 &.,'\u2019\-]"      # the characters of `DataClerk.find_companies`'s phrase rule (a test reads that source for this exact text): 1 to 60 of them
CLERK_PHRASE = re.compile(CLERK_PHRASE_CHARS + "{1,60}")
QUOTED = re.compile(r"[\"\u201c]([^\"\u201c\u201d]+)[\"\u201d]")      # a phrase in double quotes, straight or curly
THEME_NOUNS = ("stocks", "companies", "shares")      # a theme that already ends in one of these is not followed by "companies" (`TradeIntent.what`)
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
BUY_CUE = re.compile(r"\b(?:buy|buying|bought|purchase|purchasing|invest|investing|acquire|get|trade|trading|spend|spending|own|add|(?:put|place)\b[^.]{0,40}\binto|pick up|load up|go long)\b", re.I)
EACH_CUE = re.compile(r"\b(?:each|apiece|a piece|per (?:stock|company|name|ticker|symbol)s?|for every (?:one|stock|company|name|ticker|symbol))\b", re.I)      # the words that say ONE amount is for every stock (entry 125): the only word list the money needs; it filters no stock, side or phrase
PLAIN_TICKER = re.compile(r"\$?[A-Z0-9][A-Z0-9.\-]{0,9}")      # a ticker in capitals (the same shape as `symbols.is_plain_ticker`)
SELL_CUE = re.compile(r"\b(?:sell|selling|sold|unload|dump|liquidate|trim|exit|get rid of)\b", re.I)
EVEN_CUE = re.compile(r"\b(?:even|evenly|equal|equally|balanc\w*|split|divid\w*|same (?:amount|dollar\w*|money))\b", re.I)
MONEY = re.compile(r"\$?\s?(\d[\d,]*(?:\.\d+)?)\s*(k\b|thousand\b|m\b|million\b)?", re.I)
# What the user says when they have already said HOW MUCH to buy (design choices entry 85): a percentage, a dollar amount, a number of shares, or "as much as I can". The Account Manager asks how much only when none of
# these is in the request. It is a program's check of the user's own words, never a model's.
_HOW_MUCH_NOT_MONEY = (r"\d[\d,]*(?:\.\d+)?\s?(?:%|percent\b|per cent\b)"                                                                # 20%, 20 percent
                       r"|\b(?:\d[\d,]*(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)\s+shares?\b"                         # 10 shares, 0.5 shares
                       r"|\bas (?:much|many)\b|\ball[- ]in\b|\beverything\b|\bfully invest\w*"                                         # as much as I can
                       r"|\ball (?:of )?(?:my|the|our)\b[^.]{0,20}\b(?:money|cash|funds|account)"                                          # all of my money
                       r"|\b(?:the )?(?:most|max(?:imum)?)\s+(?:amount|shares?|stocks?|money)\b")                                         # the most amount, the maximum shares
_HOW_MUCH_MONEY = (r"\$\s?\d"                                                                                                          # $3,000
                   r"|\d[\d,]*(?:\.\d+)?\s*(?:k|thousand|million)?\s*(?:dollars?|bucks?|usd)\b")                                    # 3000 dollars
HOW_MUCH = re.compile("(?:" + _HOW_MUCH_NOT_MONEY + "|" + _HOW_MUCH_MONEY + ")", re.I)
HOW_MUCH_NOT_MONEY = re.compile("(?:" + _HOW_MUCH_NOT_MONEY + ")", re.I)      # everything of HOW_MUCH but money: money is read from `money_amounts`, which leaves out a price limit ("under $50 a share" is not an amount to spend)


def _at_most_the_cap(value: Decimal) -> Decimal:
    if value > MAX_BUDGET:
        raise ValueError(f"at most ${MAX_BUDGET:,.0f}")
    return value


Budget = Annotated[PositiveMoney, AfterValidator(_at_most_the_cap)]


class DirectOrder(BaseModel):
    """One trade the user named: `buy` or `sell`, a number of shares (None when they wrote none: "I want to buy MSFT"; the Account Manager then asks them in Ask mode, or works it out, design choices entry 98; a whole number, or a
    decimal of at most six places the user typed: entry 110) or the dollars they wrote for the stock (`dollars`, entry 122: "Buy $3,000 of XOM"; never both), and the stock either as a ticker (`symbol`) or as a company's name (`company`; the Data Clerk finds its symbol: entry 88). Exactly one of the two."""
    model_config = ConfigDict(extra="forbid")

    side: Literal["buy", "sell"]
    symbol: str | None = None
    company: str | None = None
    shares: sh.PositiveShares | None = None          # a Decimal: whole (no decimals) or with up to 6 decimals when the user typed a decimal
    dollars: Budget | None = None                    # the money the user wrote for THIS stock ("Buy $3,000 of XOM"; entry 122): a Decimal to the cent, above zero and at most MAX_BUDGET; shares or dollars or neither, never both
    max_price: Price | None = None                   # the most a share may cost for this order ("Buy 10 of MU under $50 a share"; entry 138): the request's price limit, copied here; a buy is placed only when the quote is at or under it
    min_price: Price | None = None                   # the least a share may cost for this order ("Sell 10 shares of MSFT at $600 or more"; entry 140): the request's price floor, copied here; a sale waits (Part B) until the quote is at or over it

    @field_validator("shares")
    @classmethod
    def _at_most(cls, value):
        if value is not None and value > MAX_DIRECT_SHARES:
            raise ValueError(f"at most {MAX_DIRECT_SHARES:,} shares")
        return value

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

    @model_serializer(mode="wrap")
    def _leave_out_a_price_limit_that_is_not_there(self, handler):
        """A dump has no `max_price` unless the user gave one: the research cache of the Proving Ground is keyed by the dump of the intent (`proving/engine.research_key`), and every key made before entry 138 must still be the key of the same request."""
        data = handler(self)
        for key in ("max_price", "min_price"):
            if data.get(key) is None:
                data.pop(key, None)
        return data

    @model_validator(mode="after")
    def _one_way_to_name_the_stock(self):
        if (self.symbol is None) == (self.company is None):
            raise ValueError("an order names the stock by its ticker (symbol) or by its company name (company), one of them")
        if self.shares is not None and self.dollars is not None:
            raise ValueError("an order has a number of shares or an amount of dollars, never both: write the one the person wrote and null for the other")
        return self

    @property
    def named(self) -> str:
        """What the user called the stock: the ticker, or the company's name as typed."""
        return self.symbol or self.company

    def describe(self) -> str:
        if self.dollars is not None:
            return f"{self.side} ${self.dollars:,.2f} of {self.named}"
        if self.shares is None:
            return f"{self.side} {self.named} (you did not say how many shares)"
        return f"{self.side} {sh.text(self.shares)} share{'s' if sh.quantity(self.shares) != 1 else ''} of {self.named}"


class TradeIntent(BaseModel):
    """The user's request as the Account Manager understood it. `request` is what they typed (kept for the record and the report)."""
    model_config = ConfigDict(extra="forbid")

    request: str
    action: Literal["buy", "research", "order", "news"]      # "order": the user named the trades themselves (`orders`): a direct order, entry 87. "news": the user only wants to SEE the news of the stocks they named (entry 149): nothing is bought, decided or traded
    topic: str | None = None
    symbols: list[str] = Field(default_factory=list)
    companies: list[str] = Field(default_factory=list)
    count: int | None = Field(default=None, ge=1, le=MAX_COUNT)
    allocation: Literal["even", "default"] = "default"      # "even": the user asked to split the money equally; "default": no split was asked (the Account Manager's own way for the mode: round-robin in Auto, what the limits allow in Ask; it was called "advisor" before entry 80, when the Advisor sized)
    budget: Budget | None = None      # the dollars the user typed, a Decimal to the cent, above zero and at most MAX_BUDGET. With `each` it is the amount for EACH purchase (the recommendation path), not a total
    each: StrictBool = False      # ONE amount written once is the amount of EVERY stock (entry 125: "$3,000 each of XOM and MSFT", "the top 3 with $3,000 each"); a bool, never a word. Code checks the user's words (`check_each`)
    max_price: Price | None = None      # a PRICE LIMIT, the most one share may cost ("under $50 a share"; entry 138): a ceiling, never an amount to spend; code reads it from the user's words (`written_amounts`)
    min_price: Price | None = None      # a PRICE FLOOR, the least one share may cost ("over $20 a share", "at $600 or more"; entry 140): never an amount to spend; code reads it from the user's words (`written_amounts`)
    shares_of: dict[str, sh.PositiveShares] = Field(default_factory=dict)      # a number of shares typed WITH a recommendation ("buy 10 of MU if the recommendation passes": {"MU": 10}; entry 140, C1), keyed by the stock as the user named it (a ticker, or a company's name as typed); a program reads it from the words
    shares_each: sh.PositiveShares | None = None      # ONE number of shares for every approved purchase ("buy 5 shares of each of the top 3 oil stocks if recommended": 5; entry 140, C1); a program reads it from the words (`SHARES_EACH`)
    raise_funds: StrictBool = False      # the user's words let the Account Manager SELL holdings to pay for a purchase the cash does not cover ("buy 2 ANT and raise funds if needed"; entry 142, R2): a program reads it from the words (`raise_funds_in`), the model only agrees
    unclear_number: Digits | None = None      # a number written with no $ and no word dollars after a money verb ("spend 1000 on MU"): shares or dollars? The desk asks (Ask mode) or stops (Auto mode) before anything else (entry 138, D5); None once settled
    corrected: list[str] = Field(default_factory=list)      # what code changed in the model's reading, one sentence each (a number the model put in the wrong kind, a price limit it missed or a minimum price that is ignored: entry 138)
    orders: list[DirectOrder] = Field(default_factory=list, max_length=MAX_COUNT)      # only with action "order"
    news_days: int | None = Field(default=None, ge=1, le=MAX_NEWS_DAYS)      # only with action "news" (entry 149): how many days of news to show, 1 to 90; a program reads an explicit window from the words (`news_window`) and wins over the model

    @model_serializer(mode="wrap")
    def _leave_out_what_entry_138_added_when_it_is_not_there(self, handler):
        """A dump has no `max_price`, `unclear_number`, `corrected`, `min_price`, `shares_of` or `shares_each` unless the request has one: every key of the Proving Ground's research cache made before entry 138 (it hashes this dump) is
        still the key of the same request, and a saved intent loads the same."""
        data = handler(self)
        for key in ("max_price", "unclear_number", "min_price", "shares_each", "news_days"):      # `news_days` is entry 149's: left out of every request that is not a news request, so the keys of the Proving Ground's research cache do not change
            if data.get(key) is None:
                data.pop(key, None)
        if not data.get("raise_funds"):
            data.pop("raise_funds", None)      # entry 142: the same for the permission to raise funds
        for key in ("corrected", "shares_of"):
            if not data.get(key):
                data.pop(key, None)
        return data

    @model_validator(mode="after")
    def _orders_go_with_the_order_action(self):
        if self.news_days is not None and self.action != "news":
            raise ValueError("news_days: the window of the news goes with a request for the news (action news)")
        if self.action == "news":
            self._a_news_request_holds_no_trade()
        if (self.shares_of or self.shares_each is not None) and self.action != "buy":
            raise ValueError("shares_of and shares_each: a number of shares typed with a recommendation goes with buying (a direct order has its shares on the order, and nothing is bought when the person only wants to look)")
        if self.raise_funds and self.action == "research":
            raise ValueError("raise_funds: holdings are sold only to pay for a purchase (nothing is bought when the person only wants to look)")
        if self.each and self.action == "research":
            raise ValueError("each: an amount for each stock goes with buying (nothing is bought when the person only wants to look)")
        if self.each and self.action == "buy" and self.budget is None:
            raise ValueError("each: the amount for each stock goes in budget (null here: say the amount the person wrote, or set each to false)")
        if self.each and self.action == "order" and self.budget is not None:
            raise ValueError("each: with orders the amount goes on EVERY order (dollars, or shares), the same in each, and the budget stays null")
        if (self.action == "order") != bool(self.orders):
            raise ValueError("orders: a direct order has the action 'order' and at least one trade, and no other action has any")
        if self.action == "order":
            numbered = any(o.shares is not None or o.dollars is not None for o in self.orders)
            if self.budget is not None and any(o.dollars is not None for o in self.orders):
                raise ValueError("orders: one total (budget) and the dollars of a stock are two ways to say how much: put the dollars of each stock on its order and leave the budget null, or give one total and no dollars on the orders")
            if self.topic or self.symbols or self.companies or self.count is not None or ((self.budget is not None or self.allocation != "default") and numbered):
                raise ValueError("orders: a direct order names the trades and nothing else (one total of dollars to spend and an even split only go with stocks named without a number of shares or dollars): ask for recommendations in a separate request")
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

    def _a_news_request_holds_no_trade(self) -> None:
        """Entry 149, N2: a request that only asks for the news holds the stocks (or a theme) and the window, and nothing that moves money or asks for a trade. Raises ValueError, with the reason, for each thing it may not hold."""
        for name, held in (("orders", bool(self.orders)), ("budget", self.budget is not None), ("each", self.each), ("count", self.count is not None), ("allocation", self.allocation != "default"), ("raise_funds", self.raise_funds),
                           ("max_price", self.max_price is not None), ("min_price", self.min_price is not None), ("shares_of", bool(self.shares_of)), ("shares_each", self.shares_each is not None)):
            if held:
                raise ValueError(f"{name}: a request for the news holds no {name.replace('_', ' ')} (nothing is bought, sold or sized): leave it empty, or, if the person also wants to buy, sell or be recommended something, this is not a news request")
        named = [*self.symbols, *self.companies]
        if self.topic and named:
            raise ValueError("a news request is for named stocks or for a theme, not both: ask for one of them at a time")
        most = load_config().news.max_stocks
        if len(named) > most:
            raise ValueError(f"a news request names at most {most} stocks ({len(named)} were named): ask for fewer at a time")

    def said_how_much(self) -> bool:
        """Whether the user's own words already say how much to buy: a dollar amount (the budget), a percentage, a number of shares, or "as much as I can" and its kin. When they do not, the Account Manager asks in Ask
        mode (`clarify.py`). A check of the typed words by a program, with no model."""
        if self.action == "order":
            return all(o.shares is not None or o.dollars is not None for o in self.orders) or self.budget is not None or bool(AS_MUCH.search(self.request))      # an order with no number is asked about unless the user said dollars (a total, or an amount for each stock: entry 122) or "as much as I can"
        named = [*self.symbols, *self.companies]
        every_stock_has_its_shares = bool(named) and all(n in self.shares_of for n in named)      # "buy 10 of MU and 5 of AMD if recommended" (entry 140, C3): the user said how much of every stock they named
        return (self.budget is not None or self.shares_each is not None or every_stock_has_its_shares or bool(HOW_MUCH_NOT_MONEY.search(self.request))
                or bool(money_amounts(self.request)))      # the dollars written are read by `money_amounts`: a price limit or a floor ("under $50 a share", "over $20 a share") is not an amount to spend (entries 138 and 140)

    @property
    def total_budget(self) -> Decimal | None:
        """The ONE total of dollars the user gave (`budget`), or None: with `each` the budget is an amount for each purchase, not a total."""
        return None if self.each else self.budget

    @property
    def each_dollars(self) -> Decimal | None:
        """The amount the user gave for EACH purchase of a request for recommendations ("the top 3 oil stocks with $3,000 each": 3000), or None. A direct order's amounts are on its orders (`DirectOrder.dollars`)."""
        return self.budget if self.each and self.action != "order" else None

    def what(self) -> str:
        """What is to be bought or looked at, in words: the theme, or the names, or the tickers. ONE company asked for (count 1, entry 146) is said in the singular."""
        if self.action == "order":
            return ", ".join(o.named for o in self.orders)
        if self.action == "news" and self.topic:
            return repr(self.topic)      # the news about a theme is searched as a term: "the news about 'quantum computing'", never 'quantum computing' companies
        if self.topic:
            ends_in_a_noun = _token_list(self.topic)[-1:] and _token_list(self.topic)[-1] in THEME_NOUNS
            return repr(self.topic) if ends_in_a_noun else f"{self.topic!r} compan{'y' if self.count == 1 else 'ies'}"      # 'AI stocks', not 'AI stocks' companies
        return ", ".join([*self.symbols, *self.companies])

    def limit_words(self) -> str:
        """The price limit and floor in words, or "" ("only while a share costs $50.00 or less", "... $20.00 or more", "... between $20.00 and $60.00")."""
        if self.max_price is not None and self.min_price is not None:
            return f"only while a share costs between ${self.min_price:,.2f} and ${self.max_price:,.2f}"
        if self.max_price is not None:
            return f"only while a share costs ${self.max_price:,.2f} or less"
        return "" if self.min_price is None else f"only while a share costs ${self.min_price:,.2f} or more"

    def shares_words(self) -> str:
        """The number of shares typed with a recommendation, in words ("10 shares of MU and 5 shares of AMD", "5 shares of each"), or ""."""
        if self.shares_each is not None:
            others = "".join(f", {sh.text(n)} of {name}" for name, n in self.shares_of.items())
            return f"{sh.text(self.shares_each)} share{'s' if sh.quantity(self.shares_each) != 1 else ''} of each" + others
        return " and ".join(f"{sh.text(n)} share{'s' if sh.quantity(n) != 1 else ''} of {name}" for name, n in self.shares_of.items())

    def describe(self) -> str:
        """One plain sentence of what was understood, for the user to confirm before anything runs."""
        return self._describe() + (" (you said to raise funds if needed: I may sell some of your holdings to pay for it)" if self.raise_funds else "") + "".join(f" (note: {note})" for note in self.corrected) + (f" (still to settle: is {canonical(self.unclear_number):,f} shares or dollars?)" if self.unclear_number is not None else "")

    def _describe(self) -> str:
        limit = self.limit_words()
        if self.action == "news":      # entry 149: only the news is shown
            days = self.news_days if self.news_days is not None else load_config().news.default_days
            return f"show the news {'about' if self.topic else 'of'} {self.what()} for the last {'day' if days == 1 else f'{days} days'} (nothing will be bought)"
        if self.action == "order":
            spend = f", spending ${self.budget:,.2f}" + (" spread evenly" if self.allocation == "even" else "") if self.budget else ""
            if self.each:      # entry 125: one amount, written once, for every stock
                one = self.orders[0]
                spend = f", spending at most ${one.dollars:,.2f} on each stock" if one.dollars is not None else f", {sh.text(one.shares)} share{'s' if sh.quantity(one.shares) != 1 else ''} of each stock"
            return ", ".join(o.describe() for o in self.orders) + spend + (f", {limit}" if limit else "") + " (your own order: no recommendation is made and the Critic is not asked)"
        verb = "buy" if self.action == "buy" else "look at (nothing will be bought)"
        count = f"the {self.count} most recommended " if self.count and self.count != 1 else "the most recommended "      # count 1: "the most recommended 'shoe' company" (entry 146)
        spread = "; the money spread evenly over the companies that are approved" if self.allocation == "even" and not self.each else ""
        spend = (f"; spending at most ${self.budget:,.2f} on each stock" if self.each else f"; spending at most ${self.budget:,.2f}") if self.budget else "; spending as much as the account allows" if self.action == "buy" else ""
        if self.shares_words():
            spend = f"; if approved, {sh.text(self.shares_each)} share{'s' if sh.quantity(self.shares_each) != 1 else ''} of it" if self.count == 1 and self.shares_each is not None and not self.shares_of else f"; if approved, {self.shares_words()}" + (f", spending at most ${self.budget:,.2f}" if self.budget else "")      # the typed shares are each stock's limit (entry 140, C2); the cash is still the only other check
        return f"{verb} {count}{self.what()}{spread}{spend}" + (f"; {limit}" if limit else "")

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


def _numbers(text: str) -> set[Decimal]:
    """Every number in the text, as a Decimal read from its digits: ($5,000, 5k, 2.5 million) and number words (three)."""
    found = set()
    for match in MONEY.finditer(text):
        value = Decimal(match.group(1).replace(",", ""))
        unit = (match.group(2) or "").lower()
        found.add(value * (1000 if unit in ("k", "thousand") else 1_000_000 if unit in ("m", "million") else 1))
    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", text, re.I):
            found.add(Decimal(value))
    return found


AS_MUCH = re.compile(r"\bas (?:much|many)\b|\ball[- ]in\b|\beverything\b|\bfully invest\w*|\ball (?:of )?(?:my|the|our)\b[^.]{0,20}\b(?:money|cash|funds|account)", re.I)      # "as much as I can" and its kin
# What makes a request one for RECOMMENDATIONS and not a direct order (entry 98: a named stock is the user's own trade unless they ask for a recommendation): checked by a program in the user's own words.
RECOMMEND_CUE = re.compile(r"\b(?:recommend\w*|advi[cs]e\w*|research\w*|analy[sz]\w*|should i|worth (?:buying|it)|if (?:the |it |they |that )?(?:pass\w*|approv\w*|looks? good|is good|are good))\b", re.I)

# "SKIP RECOMMENDATIONS" (entry 138, S1). A recommend cue the user NEGATED asks for nothing: a negating word, then within three words a word that names what is skipped (recommendations, advice, the advisor, research, analysis, the
# critic). The words between may not end the clause (no comma, no "but", "and", "or", "if" ...): "no cash but recommend three" does not negate. `not` negates only "not a recommendation" and its forms.
NEGATING = r"(?:skip(?:s|ping|ped)?|no|without|(?:don['’]?t|do not) (?:need|want)|never ?mind|forget|bypass(?:es|ing|ed)?)"
SKIPPED = r"(?:recommend\w*|advi[cs]e\w*|advisors?|research\w*|analy[sz]\w*|critic)"
BETWEEN = r"(?:\s+(?!(?:but|and|or|then|if|so|because|however|except|unless)\b)[\w'’]+){0,3}"
NO_RECOMMEND = re.compile(rf"\b(?:{NEGATING}\b{BETWEEN}|not(?:\s+(?:a|any))?)\s+{SKIPPED}\b", re.I)


def asks_for_recommendation(typed: str) -> bool:
    """Whether the user's words ask for a recommendation (or make the trade depend on one): a `RECOMMEND_CUE` outside every negated span (`NO_RECOMMEND`). "Buy $2500 worth of KO and skip recommendations." does not ask for one;
    "Buy 10 of MU if the recommendation passes" does. A program's check of the user's own words, the one reader of the cue."""
    negated = [match.span() for match in NO_RECOMMEND.finditer(typed)]
    return any(not any(a <= cue.start() and cue.end() <= b for a, b in negated) for cue in RECOMMEND_CUE.finditer(typed))


def skips_recommendations(typed: str) -> bool:
    """Whether the user said to skip the recommendations and asks for none anywhere else in their words."""
    return bool(NO_RECOMMEND.search(typed)) and not asks_for_recommendation(typed)


# PERMISSION TO RAISE FUNDS (entry 142, R2): a closed list of phrases a PROGRAM reads in the user's own words, and the negations that win over it. It decides whether holdings may be SOLD, so the model does not.
_SELL_WHAT = r"(?:something|some|whatever\s+(?:you|i)\s+need|what\s+(?:you|i)\s+need|(?:(?:some|any|my|other|a few)\s+)?(?:holdings?|shares?|stocks?|positions?))"
RAISE_FUNDS_CUE = re.compile(r"\b(?:raise\s+(?:the\s+|some\s+|more\s+|enough\s+)?(?:funds|cash|money)"
                             r"|free(?:\s+up|ing\s+up)\s+(?:the\s+|some\s+|more\s+|enough\s+)?(?:cash|money|funds)"
                             rf"|sell\s+{_SELL_WHAT}\s+(?:to|in order to)\s+(?:pay|cover|fund|buy|afford)"
                             r"|take\s+(?:some\s+)?profits?\s+(?:to|in order to)\s+(?:pay|buy|fund|cover))\b", re.I)
NO_RAISE_FUNDS = re.compile(r"\b(?:(?:don['’]?t|do not|never|without|no need to|not to)\s+(?:(?:you|i|please|ever|any)\s+)?(?:sell\w*|rais\w*\s+(?:the\s+|any\s+)?(?:funds|cash|money)|free\w*\s+up|tak\w*\s+(?:any\s+)?profits?)|no\s+(?:selling|sales))\b", re.I)


def raise_funds_in(typed: str) -> bool:
    """Whether the user's words let the Account Manager sell holdings to pay for the purchase: a `RAISE_FUNDS_CUE` and no `NO_RAISE_FUNDS` anywhere in the words. "Buy 2 shares of ANT and raise funds if needed" does;
    "buy 2 ANT but don't sell anything" and "buy 2 ANT, don't raise funds" do not. A program's check of the user's own words, the one reader of the cue."""
    return bool(RAISE_FUNDS_CUE.search(typed)) and not NO_RAISE_FUNDS.search(typed)


def denies_raise_funds(typed: str) -> bool:
    """Whether the user said NOT to sell anything ("don't sell anything", "without raising funds"): it wins over the setting ALWAYS_RAISE_FUNDS for that request (entry 142, R12)."""
    return bool(NO_RAISE_FUNDS.search(typed))


class NeedsClarification(ValueError):
    """The request cannot be served as written and no second try of the model can fix it (entry 138): the user has to say more. `AccountManager.understand_request` raises it at once, with no retry; its message is for the user."""


SKIP_NEEDS_STOCKS = "You asked me to skip the recommendations, but no stock to buy is named: only the recommendations choose stocks for a theme. Name the stocks to buy (for example 'Buy $2,500 of ABCD and skip recommendations'), or let me recommend some."      # S3


numbers_in = _numbers      # the public name: `clarify.py` checks the amount of an answer against the numbers the user typed

# MONEY is a number the user WROTE AS MONEY (design choices entries 122 and 138): a `$` before it or the word dollars (bucks, usd) after it, with or without a unit of thousands or millions ("$5k", "5 thousand dollars"). A bare number
# is not money, and neither is `3k` or `2 million` alone, a number of shares or a percentage: without a `$` or the word dollars a number is a quantity (Jacob, 2026-11-20). The one other thing that writes a number as money is the money
# it is said to be: "I have 10000 to spend", "10000 in cash" (the words `to spend`, `to invest`, `in cash`, `of cash`, `in funds` right after it): Jacob's own request of 2026-11-15 says it that way.
MONEY_WRITTEN = re.compile(r"(?P<sign>\$\s?)?(?P<n>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>k\b|thousand\b|m\b|million\b)?(?P<word>\s*(?:dollars?|bucks?|usd)\b|\s*(?:to (?:spend|invest)|in (?:cash|funds)|of cash)\b)?", re.I)
# A PRICE LIMIT (entry 138, L1): a money amount tied to ONE share ("a share", "per share", "each share", "/share", "a pop"), or after a word that caps a price (under, below, less than, cheaper than, priced under) and not followed by
# "of", "worth", "in" a stock (that is an amount to spend). "at most $5,000" and "no more than $5,000" with no share beside them are amounts to spend. A word that sets a floor (over, above, more than, at least) before the amount, or
# "or more", "or higher" after it, is a PRICE FLOOR (entry 140; "or less", "or lower" after it is a ceiling), and "between $20 and $60 a share" is both. "or better" is a floor for a SALE and a ceiling for a purchase: a program reads the side
# from the last buying or selling word before the amount, and with none it is a purchase (a recommendation is a purchase).
TIED_TO_A_SHARE = re.compile(r"\s*(?:(?:a|per|each|every)\s+share\b|/\s*share\b|(?:a|per)\s+pop\b)", re.I)
CEILING_WORD = re.compile(r"(?:under|below|less than|cheaper than|priced under)\s*$", re.I)
FLOOR_WORD = re.compile(r"(?:over|above|(?<!no )(?<!not )more than|at least|greater than|no less than)\s*$", re.I)      # "no more than $50 a share" is a ceiling, not a floor
TO_SPEND = re.compile(r"\s*(?:of|worth|in|on|into)\b", re.I)
OR_WORD = re.compile(r"\s*or\s+(?P<w>more|higher|better|less|lower)\b", re.I)      # "$600 or more", "$90 or less" (entry 140)
PRICE_CONTEXT_BEFORE = re.compile(r"(?:\bat|@|\bpriced at|\b(?:price|limit)(?:\s+(?:is|of|at|to be))?)\s*$", re.I)      # what stands before an amount with no share beside it when "or more" / "or less" after it makes it a price: "at $600 or more", "if the price is $90 or less" ("with $1,000 or more" is money)
SPEND_VERB_BEFORE = re.compile(r"\b(?:spend\w*|invest\w*|budget\w*|allocat\w+|deploy\w*|put(?:ting)?|place|placing)\b(?:\s+(?:about|around|roughly|approximately|up to|at most|some|a total of|only|just))?\s*$", re.I)      # "invest $5,000 or less": an amount to spend, not a price
BETWEEN_OPEN = re.compile(r"\bbetween\s*$", re.I)
BETWEEN_GAP = re.compile(r"\s*(?:(?:a|per|each|every)\s+share\s*)?(?:and|to)\s*$", re.I)      # "between $20 and $60 a share": what stands between the two amounts


@dataclass(frozen=True)
class Written:
    """One amount the user wrote with a `$` or the word dollars: its `value`, where it is in the text, and what it is: `money` (an amount to spend), `ceiling` (a price limit) or `floor` (a minimum price, entry 140)."""
    value: Decimal
    start: int
    end: int
    kind: Literal["money", "ceiling", "floor"]


def written_amounts(typed: str) -> list[Written]:
    """Every amount the user's words write as money, in order, each with its kind (see `MONEY_WRITTEN` and `TIED_TO_A_SHARE`): the ONE place that says whether `$50` is an amount to spend or a price per share. A program's check of the
    user's own words, no model."""
    found = []
    for match in MONEY_WRITTEN.finditer(typed):
        if not (match["sign"] or match["word"]):
            continue
        unit = (match["unit"] or "").lower()
        value = Decimal(match["n"].replace(",", "")) * (1000 if unit in ("k", "thousand") else 1_000_000 if unit in ("m", "million") else 1)
        before, after = typed[:match.start()], typed[match.end():]
        tie = TIED_TO_A_SHARE.match(after)
        tied = bool(tie)
        or_word = OR_WORD.match(after[tie.end():] if tie else after)      # "$600 or more", "$20 a share or more"
        spent = bool(TO_SPEND.match(after[tie.end() + or_word.end():] if tie and or_word else after[or_word.end():] if or_word else after))
        if or_word and (spent or (not tied and (SPEND_VERB_BEFORE.search(before) or not PRICE_CONTEXT_BEFORE.search(before)))):
            kind = "money"      # "$5,000 or more of MU", "invest $5,000 or less", "with $1,000 or more": an amount to spend (with no share beside it, only "at", "price" or "limit" before it makes it a price)
        elif or_word:
            word = or_word["w"].lower()
            kind = "floor" if word in ("more", "higher") else "ceiling" if word in ("less", "lower") else "floor" if _cue_before(typed, match.start()) == "sell" else "ceiling"
        elif FLOOR_WORD.search(before) and (tied or not spent):
            kind = "floor"
        elif tied or (CEILING_WORD.search(before) and not spent):
            kind = "ceiling"
        else:
            kind = "money"
        found.append(Written(value, match.start(), match.end(), kind))
    for first, second in zip(found, found[1:]):      # "between $20 and $60 a share": the lower amount is the floor and the higher the ceiling, when the second is no amount to spend ("between $20 and $60 of MU")
        gap = typed[first.end:second.start]
        opened = BETWEEN_OPEN.search(typed[:first.start])
        if (opened and first.kind in ("money", "ceiling") and second.kind in ("money", "ceiling") and "ceiling" in (first.kind, second.kind) and BETWEEN_GAP.fullmatch(gap)      # at least one is tied to a share
                and not SPEND_VERB_BEFORE.search(typed[:opened.start()]) and not TO_SPEND.match(typed[second.end:])):
            low, high = sorted((first, second), key=lambda w: w.value)
            found[found.index(first)] = Written(first.value, first.start, first.end, "floor" if first is low else "ceiling")
            found[found.index(second)] = Written(second.value, second.start, second.end, "floor" if second is low else "ceiling")
    return found


def money_amounts(typed: str, strict: bool = False) -> list[Decimal]:
    """Every amount the user's words write as MONEY to spend, each time it is written, in order, as a Decimal read from its digits ("$3,000", "3,000 dollars", "$5k", "2.5 million dollars"). A bare number, `3k` alone, a number written
    as shares or a percentage, and a PRICE LIMIT ("under $50 a share": `written_amounts`) are not in it. `strict` is kept for the callers of entry 122: since entry 138 every amount needs the `$` or the word dollars. A program's check of
    the user's own words, no model."""
    return [w.value for w in written_amounts(typed) if w.kind == "money"]


def money_written(typed: str, strict: bool = False) -> set[Decimal]:
    """The amounts of `money_amounts`, each once."""
    return set(money_amounts(typed, strict))


def price_limits(typed: str) -> list[Decimal]:
    """The price limits (ceilings) the user's words write, each once, in order of appearance ("under $50 a share": 50)."""
    return list(dict.fromkeys(w.value for w in written_amounts(typed) if w.kind == "ceiling"))


def price_floors(typed: str) -> list[Decimal]:
    """The price floors (the least one share may cost) the user's words write, each once, in order of appearance ("over $20 a share", "at $600 or more": entry 140)."""
    return list(dict.fromkeys(w.value for w in written_amounts(typed) if w.kind == "floor"))


MONEY_VERB_NUMBER = re.compile(r"\b(?:spend(?:ing)?|invest(?:ing)?|put(?:ting)?|place|placing|allocat\w+|deploy\w*|pour\w*)\s+(?:about |around |roughly |approximately |up to |at most |some )?"
                               r"(?P<n>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>k\b|thousand\b|m\b|million\b)?(?![\d.,])"
                               r"(?!\s*(?:%|percent\b|per ?cent\b|shares?\b|dollars?\b|bucks?\b|usd\b|stocks?\b|compan\w+|names?\b|picks?\b|ideas?\b))", re.I)


def ambiguous_numbers(typed: str) -> list[tuple[Decimal, int]]:
    """The numbers written right after a money verb (spend, invest, put, place, allocate ...) with no `$`, no word dollars, no `shares`, no `%`: "spend 1000 on MU" (entry 138, D5). Each is (the value, where it starts). Without a `$` or
    the word dollars a number is a quantity, but after "spend" the person may well have meant dollars, and buying 1000 shares of a stock with real money when $1,000 was meant is the one mistake that cannot be undone: such a number is
    never read either way without asking. A program's check of the user's own words."""
    out = []
    for match in MONEY_VERB_NUMBER.finditer(typed):
        unit = (match["unit"] or "").lower()
        out.append((Decimal(match["n"].replace(",", "")) * (1000 if unit in ("k", "thousand") else 1_000_000 if unit in ("m", "million") else 1), match.start("n")))
    return out


def check_dollars(order: DirectOrder, typed: str) -> None:
    """Raises ValueError, with the reason, unless the dollars of this order are an amount the user wrote as money (entry 122, rule D2): the number is checked, and no word around the stock is. An order with no dollars passes."""
    if order.dollars is None or order.dollars in money_written(typed):
        return
    amount = f"{canonical(order.dollars):f}"
    if order.dollars in _numbers(typed):
        raise ValueError(f"the amount {amount} is in what you typed, but not as money (a dollar sign, or the word dollars, bucks or usd): it may be a number of shares or a percentage, which are not dollars")
    raise ValueError(f"the amount {amount} is not in what you typed: the dollars of {order.named} must be an amount the person wrote as money")


def check_dollars_each(intent: TradeIntent, typed: str) -> None:
    """Raises ValueError, with the reason, when an amount is the dollars of more orders than the user wrote it for (entry 122): "Buy $5,000 of MU and AMD" writes $5,000 once, so it is ONE total to spend (the budget), not $5,000 for each
    of two stocks, which would spend $10,000 on a request for $5,000. The amounts are counted, a number against a number; no word of the request is read. An order with no dollars passes."""
    if intent.each:      # ONE amount for every stock, said so (entry 125): `check_each` has the words and the numbers
        return
    written = Counter(money_amounts(typed))
    for amount, orders in Counter(o.dollars for o in intent.orders if o.dollars is not None).items():
        if orders > written[amount]:
            raise ValueError(f"the amount {canonical(amount):f} is the dollars of {orders} orders but you wrote it {written[amount]} time{'s' if written[amount] != 1 else ''}: an amount written once for several stocks is ONE total to spend (the budget, with no dollars on the orders); "
                             "it is the dollars of a stock only when the person wrote an amount for that stock, or said each (each, apiece, per stock: then set each to true)")


def shares_written(typed: str) -> set[Decimal]:
    """Every number the user wrote as a number of SHARES ("10 shares", "0.5 shares", "three shares"), as a Decimal read from its digits: a number with more than six decimals is left out (it is not a quantity: `shares.PLACES`). A program's check of the user's own words."""
    words = "|".join(NUMBER_WORDS)
    found = set()
    for match in re.finditer(rf"(?<![\w$.,%-])(\d[\d,]*(?:\.\d+)?|{words})(?![\d.,%])\s+shares?\b", typed, re.I):
        word = match.group(1).lower()
        written = str(NUMBER_WORDS[word]) if word in NUMBER_WORDS else word.replace(",", "")
        if len(written.partition(".")[2]) <= sh.PLACES:
            found.add(sh.quantity(Decimal(written)))
    return found


def check_each(intent: TradeIntent, typed: str) -> None:
    """Raises ValueError, with the reason, unless an `each` the model read is what the user said (entry 125). ONE amount may be the amount of every stock only when (a) a word that says each is in the user's text (`EACH_CUE`: each,
    apiece, a piece, per stock ...: a small closed list, because this decides how much money is spent), (b) every order carries that same amount, the dollars of every order or the shares of every order (the budget stays null), or, for
    a request for recommendations, the budget is that amount, and (c) it is an amount the user wrote as money (shares are checked by `check_order` as a number of shares). Without `each` this does nothing."""
    if not intent.each:
        return
    if not EACH_CUE.search(typed):
        raise ValueError("each is true but nothing you typed says each (each, apiece, a piece, per stock, per company, for every one): an amount written once for several stocks is ONE total (the budget, each false)")
    if intent.action == "order":
        dollars, count = {o.dollars for o in intent.orders}, {o.shares for o in intent.orders}
        if not ((None not in dollars and len(dollars) == 1 and count == {None}) or (None not in count and len(count) == 1 and dollars == {None})):
            raise ValueError("each means ONE amount for every stock: put the same amount on EVERY order (the same dollars, or the same shares) and leave the budget null")
        return
    if intent.budget not in money_written(typed):
        raise ValueError(f"the amount {canonical(intent.budget):f} for each stock is not in what you typed as money (a dollar sign, or the word dollars, bucks or usd)")


def check_amount_was_read(intent: TradeIntent, typed: str) -> None:
    """Raises ValueError, with the reason, when the user wrote an amount of money (a `$` or the word dollars) and the reading of a direct order has no budget, no dollars on any order and no shares on any order (entry 122, rule D5):
    the Account Manager never falls back to spending all the cash on a request that said an amount. A request with no such amount keeps the older ways (Ask mode asks how much; Auto spends the cash round robin)."""
    if intent.action != "order" or intent.budget is not None or any(o.dollars is not None or o.shares is not None for o in intent.orders):
        return
    amounts = sorted(money_written(typed, strict=True))
    if amounts:
        raise ValueError(f"you wrote an amount of money (${canonical(amounts[0]):,f}) but no amount was read for any stock: put the dollars of each stock on its order (dollars), or give one total to spend (budget)")


def check_budget(intent: TradeIntent, typed: str) -> None:
    """Raises ValueError, with the reason, unless the budget (the ONE total of dollars) is an amount the user wrote as money (entry 138): a `$` before it or the word dollars after it. A bare number is a quantity, never a budget: it is
    taken for shares before this check where the user's words give it as shares (`_read_numbers`), and otherwise refused here. With `each` the check is `check_each`'s. A request with no budget passes."""
    if intent.budget is None or intent.each or intent.budget in money_written(typed):
        return
    amount = f"{canonical(intent.budget):f}"
    if intent.budget in _numbers(typed):
        raise ValueError(f"the amount {amount} is in what you typed, but not as money (a $ sign or the word dollars, bucks, usd): without them it is a quantity, not dollars. Leave the budget null, and put a number of shares on the order it belongs to")
    raise ValueError(f"the amount {amount} is not in what you typed")


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 3}


def _token_list(text: str) -> list[str]:
    """Every word of the text, in order, lower case: letters and digits ("5G" is "5g")."""
    return re.findall(r"[a-z0-9]+", text.lower())


def theme_is_in(topic: str, typed: str) -> bool:
    """Whether the theme is the user's own words (entry 113). A theme with a word of three letters or more passes when it shares such a word with what was typed (the old check, unchanged). A theme made only of shorter
    words (AI, EV, 5G, IT) has no such word to compare, so it passes when EVERY one of its words is a whole word of what was typed; a short word that is not in the text never passes."""
    long = _words(topic)
    if long:
        return bool(long & _words(typed))
    every = set(_token_list(topic))
    return bool(every) and every <= set(_token_list(typed))


def quoted_theme(topic, typed: str) -> str | None:
    """The phrase the user put in double quotes that IS their theme, or None (entry 113). It is the one quoted phrase that holds every word of the model's topic, as typed with the spaces collapsed ("AI stocks" for the topic
    `AI`), and only when it passes the topic's own check (`NAME`) and the Data Clerk's phrase rule (`CLERK_PHRASE`). Two different quoted phrases that both hold the topic's words are no answer, and neither is a topic with no
    word in it. The caller decides when it applies: a search with a topic, no ticker and no company."""
    wanted = set(_token_list(topic)) if isinstance(topic, str) else set()
    if not wanted:
        return None
    phrases = {}
    for match in QUOTED.finditer(typed):
        phrase = " ".join(match.group(1).split())
        if phrase and wanted <= set(_token_list(phrase)):
            phrases.setdefault(phrase.casefold(), phrase)
    if len(phrases) != 1:
        return None
    phrase = next(iter(phrases.values()))
    return phrase if NAME.match(phrase) and CLERK_PHRASE.fullmatch(phrase) else None


def _cue_before(typed: str, position: int) -> str | None:
    """The side of the LAST buying or selling word before `position` in the user's words ("buy 50 MSFT, sell 10 TSLA": "buy" before the 50, "sell" before the 10), or None when there is none."""
    cues = [(m.start(), "buy") for m in BUY_CUE.finditer(typed, 0, position)] + [(m.start(), "sell") for m in SELL_CUE.finditer(typed, 0, position)]
    return max(cues)[1] if cues else None


def _order_phrases(typed: str, name: str, ticker: bool = True) -> list[tuple[int, int]]:
    """Every (number of shares, where it starts) the user wrote right before this ticker or company name: "50 shares of MSFT", "50 shares in MSFT", "50 MSFT", number words up to ten, and up to three words in between
    ("50 shares of Microsoft Corp (MSFT)"). The number may have a decimal part of at most six digits ("0.5 shares of MSFT", "81.004455 shares of DPZ": entry 110): a whole number is an int and a decimal a float, as
    `shares.quantity` gives them; a number with MORE than six decimals raises ValueError (it is not rounded: a number of shares has at most six). A dollar amount ("$5,000.50") or a percentage is not a number of shares and is
    never read as one, and neither is the last word of a hyphenated number ("twenty-five" is not "five")."""
    words = "|".join(NUMBER_WORDS)
    stock = re.escape(name).replace(r"\ ", r"\s+")
    pattern = re.compile(rf"(?<![\w$.,%-])(\d[\d,]*(?:\.\d+)?|{words})(?![\d.,%])\s+(?!(?:dollars?|bucks?|usd|thousand|million)\b)(?:shares?\s+)?(?:(?:of|in)\s+)?(?:[A-Za-z&'’-]+\s+){{0,3}}\(?\$?{stock}(?![A-Za-z0-9])", re.I)
    unclear = {at for _, at in ambiguous_numbers(typed)}      # "spend 1000 on MU": the number after a money verb is not read as shares (entry 138, D5)
    found = []
    for match in pattern.finditer(typed):
        if match.start() in unclear:
            continue
        word = match.group(1).lower()
        if word in NUMBER_WORDS:
            found.append((Decimal(NUMBER_WORDS[word]), match.start()))
            continue
        written = word.replace(",", "")
        if len(written.partition(".")[2]) > sh.PLACES:
            raise ValueError(f"{written} shares of {name} has more than {sh.PLACES} decimals: a number of shares has at most {sh.PLACES} (0.{'0' * (sh.PLACES - 1)}1 is the smallest)")
        found.append((sh.quantity(Decimal(written)), match.start()))
    return found


def check_order(order: DirectOrder, typed: str, each: bool = False) -> None:
    """Raises ValueError, with the reason, unless the user's own words give this trade: the ticker as typed (or the company's name, case ignored), the shares as a number of shares right before it, and the side of the
    buying or selling word before that. With `each` (entry 125: "Buy 10 shares each of AAPL and MSFT") the shares may be a number of shares written ONCE for every stock: it is then checked as a number of shares anywhere in the text."""
    if order.symbol is not None:
        if not re.search(rf"(?<![A-Za-z0-9]){re.escape(order.symbol)}(?![A-Za-z0-9])", typed):
            raise ValueError(f"the ticker {order.symbol} is not in what you typed")
    elif not re.search(r"\s+".join(map(re.escape, order.company.split())), typed, re.I):
        raise ValueError(f"the company {order.company!r} is not in what you typed")
    if asks_for_recommendation(typed):
        raise ValueError("you asked for recommendations or advice (or made the trade depend on them): that is not a direct order, so it is a request to buy or research")
    if order.shares is None:
        typed_numbers = _order_phrases(typed, order.named, ticker=order.symbol is not None) if order.dollars is None else []      # dollars for the stock: the amount is checked as money (`check_dollars`), not as shares
        if typed_numbers:
            raise ValueError(f"you typed {sh.text(typed_numbers[0][0])} shares of {order.named}: put the number in shares, not null")
        position = next((m.start() for m in re.finditer(r"(?<![A-Za-z0-9])" + r"\s+".join(map(re.escape, order.named.split())) + r"(?![A-Za-z0-9])", typed, 0 if order.symbol else re.I)), None)
        if position is None or _cue_before(typed, position) != order.side:
            word = "buying word (buy, purchase, get ...)" if order.side == "buy" else "selling word (sell, unload, dump ...)"
            raise ValueError(f"{order.side} {order.named} is not what you typed: the last word about buying or selling before {order.named} is not a {word}")
        return
    phrases = [at for shares, at in _order_phrases(typed, order.named, ticker=order.symbol is not None) if shares == order.shares]
    if not phrases and each and order.shares in shares_written(typed):      # one number of shares for every stock, said so: the side is the last buying or selling word before the stock
        position = next((m.start() for m in re.finditer(r"(?<![A-Za-z0-9])" + r"\s+".join(map(re.escape, order.named.split())) + r"(?![A-Za-z0-9])", typed, 0 if order.symbol else re.I)), None)
        phrases = [] if position is None or _cue_before(typed, position) != order.side else [position]
        if not phrases:
            word = "buying word (buy, purchase, get ...)" if order.side == "buy" else "selling word (sell, unload, dump ...)"
            raise ValueError(f"{order.side} {order.named} is not what you typed: the last word about buying or selling before {order.named} is not a {word}")
    if not phrases:
        raise ValueError(f"{sh.text(order.shares)} shares of {order.named} is not what you typed: say it as a number of shares, for example 'buy {sh.text(order.shares)} shares of {order.named}'")
    if not any(_cue_before(typed, at) == order.side for at in phrases):
        word = "buying word (buy, purchase, get ...)" if order.side == "buy" else "selling word (sell, unload, dump ...)"
        raise ValueError(f"{order.side} {order.named} is not what you typed: the last word about buying or selling before the {sh.text(order.shares)} shares is not a {word}")


# THE WINDOW OF A NEWS REQUEST (entry 149, N3): a PROGRAM reads an explicit span from the user's words and wins over the model. A number of days moves no money, but it is a number, so code reads it. A closed list: "last/past N days/weeks/months"
# ("last week" and "last month" with no number are 1), "this week", "this month", "today", "yesterday". A week is 7 days and a month 30. "this year" and "last year" are not read (over the limit): the model's value, or the default, stands.
NEWS_UNITS = {"day": 1, "week": 7, "month": 30}
NEWS_LAST = re.compile(rf"\b(?:last|past)\s+(?:(?P<n>\d+|{'|'.join(NUMBER_WORDS)})\s+)?(?P<unit>day|week|month)s?\b", re.I)
NEWS_THIS = re.compile(r"\bthis\s+(?P<unit>week|month)\b", re.I)
NEWS_TODAY = re.compile(r"\btoday\b", re.I)
NEWS_YESTERDAY = re.compile(r"\byesterday\b", re.I)


def news_window(typed: str) -> int | None:
    """The number of days the user's words ask the news for, or None when they write none: "last 3 days" 3, "past 2 weeks" 14, "last week" 7, "last month" 30, "this week" 7, "this month" 30, "today" 1, "yesterday" 2 (it includes today). When
    the words write more than one span ("today and the last week") the longest is the window. A span of 0 is not one. A program's check of the user's own words, no model (entry 149, N3); `news.max_days` is applied by the caller."""
    found = []
    for match in NEWS_LAST.finditer(typed):
        word = (match["n"] or "1").lower()
        found.append((NUMBER_WORDS[word] if word in NUMBER_WORDS else int(word)) * NEWS_UNITS[match["unit"].lower()])
    found += [NEWS_UNITS[match["unit"].lower()] for match in NEWS_THIS.finditer(typed)]
    found += [1] * len(NEWS_TODAY.findall(typed)) + [2] * len(NEWS_YESTERDAY.findall(typed))
    found = [days for days in found if days >= 1]
    return max(found) if found else None


def check_named(intent: TradeIntent, typed: str) -> None:
    """Raises ValueError, with the reason, unless the stocks and the theme of a request for companies (or for the news) are the user's own words: at least one of them, each ticker as typed, each company's name (case ignored) and the
    theme sharing a word with the text (`theme_is_in`)."""
    lowered = typed.lower()
    if not (intent.topic or intent.symbols or intent.companies):
        raise ValueError("no theme, ticker or company: name what you want to buy or look at")
    for symbol in intent.symbols:
        if not re.search(rf"(?<![A-Za-z0-9]){re.escape(symbol)}(?![A-Za-z0-9])", typed):
            raise ValueError(f"the ticker {symbol} is not in what you typed")
    for name in intent.companies:
        if name.lower() not in lowered:
            raise ValueError(f"the company {name!r} is not in what you typed")
    if intent.topic and not theme_is_in(intent.topic, typed):
        raise ValueError(f"the theme {intent.topic!r} shares no word with what you typed")


def _news_intent(raw: dict, typed: str) -> TradeIntent:
    """The form of a request that only asks for the news (entry 149), accepted only when the user's words support it. The window is the one the words write (`news_window`), else the model's `news_days` (a whole number of days, 1 or more),
    else `news.default_days`, cut to `news.max_days` with a sentence in `corrected` that tells the user. Nothing that moves money is read here (no price limit, no number of shares, no permission to raise funds: `_read_numbers` is not run), and
    whatever the model put in such a field is refused by the intent's own validator with its reason."""
    raw, notes, cap = dict(raw), [], load_config().news
    written, model_days = news_window(typed), raw.pop("news_days", None)
    if model_days is not None:      # a whole number of days from the model's JSON, through the door of the number (a bool and a text are refused)
        try:
            exact = digits(model_days)
        except ValueError:
            exact = None
        if exact is None or exact != exact.to_integral_value() or exact < 1:
            raise ValueError("news_days must be a whole number of days (1 or more), or null when the person gave no window")
        model_days = int(exact)
    days = written if written is not None else model_days if model_days is not None else cap.default_days
    if days > cap.max_days:
        notes.append(f"the news is kept for at most {cap.max_days} days: showing {cap.max_days}")
        days = cap.max_days
    raw["news_days"] = days
    named = [*(raw.get("symbols") or []), *(raw.get("companies") or [])]
    if len(named) > cap.max_stocks:      # the user has to say fewer: the model is not asked again (it would drop one without telling them; found by the first real capture)
        raise NeedsClarification(f"a news request names at most {cap.max_stocks} stocks and {len(named)} were named: ask for the news of {cap.max_stocks} or fewer at a time")
    if raw.pop("raise_funds", None):
        raw["raise_funds"] = True      # the intent refuses it: nothing is sold for the news
    raw.pop("unclear_number", None)
    try:
        intent = TradeIntent(request=typed, corrected=notes, **raw)
    except ValidationError as error:
        first = error.errors()[0]
        raise ValueError(f"{'.'.join(str(x) for x in first['loc']) or 'the form'}: {first['msg']}") from None
    except ValueError as error:   # check_symbol
        raise ValueError(str(error)) from None
    check_named(intent, typed)
    return intent


def check_typed(typed) -> str:
    """The user's words with the spaces tidied. Raises ValueError for nothing at all or for more than the form can read: a request refused here costs no model call."""
    typed = " ".join(typed.split()) if isinstance(typed, str) else ""
    if not typed:
        raise ValueError("there is nothing to understand: type what you want to buy or look at")
    if len(typed) > MAX_TEXT:
        raise ValueError(f"the request is longer than {MAX_TEXT} characters")
    return typed


COUNT_PHRASE = re.compile(rf"\b(?:top|best|first|biggest|largest)\s+(?P<a>\d+|{'|'.join(NUMBER_WORDS)})\b"
                          rf"|\b(?P<b>\d+|{'|'.join(NUMBER_WORDS)})\s+(?:(?:most|best|top|biggest|largest)\b|(?:\w+\s+){{0,2}}(?:stocks?|compan(?:y|ies)|names?|picks?|ideas?|tickers?|symbols?)\b)", re.I)


def count_written(typed: str) -> set[Decimal]:
    """The numbers the user wrote as a COUNT of companies (entry 138, D3): right after top, best or first, or right before most, best, top, stocks, companies, names, picks. "the top 3 AI stocks": 3. "buy 10 of MU" writes none (10 is
    a number of shares). A program's check of the user's own words."""
    found = set()
    for match in COUNT_PHRASE.finditer(typed):
        word = (match["a"] or match["b"]).lower()
        found.add(Decimal(NUMBER_WORDS[word]) if word in NUMBER_WORDS else Decimal(word))
    return found


ONE_STOCK = re.compile(r"\bthe\s+(?:single\s+|very\s+)?(?:most[\s-]+recommended|best|top|number\s+one|#\s?1|no\.?\s?1|(?:highest|best|top)[\s-]+(?:rated|ranked|scored))\s+(?:[\w&'.-]+\s+){0,3}?"
                       r"(?:stock|company|name|pick|ticker|equity)\b(?![\s-]+(?:stocks|companies|names|picks|ideas|tickers|symbols|list)\b)", re.I)      # "the most recommended shoe company", "the best AI stock": ONE company (entry 146)


def one_stock_asked(typed: str) -> bool:
    """True when the user asked for ONE company in the singular with no number ("buy 5 shares of the most recommended shoe company", "the best AI stock", "the top-rated oil company"; entry 146): the count is 1. A plural
    ("the best shoe companies", "the top stock picks") is not. A program's check of the user's own words: a count of 1 decides that one stock, not three, is bought."""
    return bool(ONE_STOCK.search(typed))


SHARES_EACH = re.compile(rf"(?<![\w$.,%-])(?P<n>\d[\d,]*(?:\.\d+)?|{'|'.join(NUMBER_WORDS)})(?![\d.,%])\s+(?:shares?\s+)?(?:(?:of|in)\s+)?"
                         r"(?:each|apiece|a piece|per\s+(?:stock|company|name|ticker|symbol)s?|for every (?:one|stock|company|name|ticker|symbol))\b", re.I)      # "5 shares of each", "5 shares each", "5 apiece": ONE number of shares for every stock (entry 140, C1)


def shares_each_written(typed: str) -> list[Decimal]:
    """The numbers the user wrote as a number of shares for EVERY stock ("5 shares of each of the top 3 oil stocks", "5 shares each": entry 140, C1), each once, as a Decimal read from its digits; a number with more than six decimals is left out.
    A money amount ("$3,000 each") is not one: its number follows a `$`. A program's check of the user's own words."""
    found = []
    for match in SHARES_EACH.finditer(typed):
        word = match["n"].lower()
        written = str(NUMBER_WORDS[word]) if word in NUMBER_WORDS else word.replace(",", "")
        if len(written.partition(".")[2]) <= sh.PLACES and sh.quantity(Decimal(written)) not in found:
            found.append(sh.quantity(Decimal(written)))
    return found


def typed_shares(intent: TradeIntent, typed: str) -> tuple[dict[str, Decimal], Decimal | None]:
    """The shares the user typed WITH a recommendation (entry 140, C1): ({the stock as the user named it: the shares written right before it}, the number of shares written for EVERY stock or None). Read from the words by the same code
    as a direct order's shares (`_order_phrases`); the count of a request ("the 3 best MU ...") is never shares. With a number for every stock a stock's own number stays only when it is a different one. The number of shares of a purchase
    that a recommendation makes is a WHOLE number (a fraction comes from an even split or a direct order and from nothing else: entry 110): a fraction, two different numbers for one stock or for every stock raise `NeedsClarification`."""
    counts = {m.start("a") if m["a"] else m.start("b") for m in COUNT_PHRASE.finditer(typed)}
    each = shares_each_written(typed)
    if len(each) > 1:
        raise NeedsClarification("you wrote more than one number of shares for every stock (" + ", ".join(sh.text(n) for n in each) + "): say one")
    of: dict[str, Decimal] = {}
    for name, is_ticker in [*((s, True) for s in intent.symbols), *((c, False) for c in intent.companies)]:
        written = list(dict.fromkeys(n for n, at in _order_phrases(typed, name, ticker=is_ticker) if at not in counts))
        if len(written) > 1:
            raise NeedsClarification(f"you wrote more than one number of shares for {name} ({', '.join(sh.text(n) for n in written)}): say one")
        if written and not (each and written[0] == each[0]):
            of[name] = written[0]
    for value in [*of.values(), *each]:
        if value != value.to_integral_value():
            raise NeedsClarification(f"{sh.text(value)} shares with a recommendation: a recommendation is bought in whole shares (a fraction of a share comes only from a direct order, or an even split). Say a whole number, or order the stock yourself with no recommendation")
    return of, each[0] if each else None


def _named(raw: dict, orders: list[dict]) -> list[tuple[str, bool]]:
    """The stocks the model's reading names, as (name as typed, is a ticker): the orders' own, else the form's symbols and companies."""
    if orders:
        return [(o["symbol"].strip().lstrip("$"), True) if "symbol" in o else (" ".join(o["company"].split()), False) for o in orders]
    return [(s.strip().lstrip("$"), True) for s in raw.get("symbols") or []] + [(" ".join(c.split()), False) for c in raw.get("companies") or []]


def _read_numbers(raw: dict, typed: str) -> tuple[dict, list[str], Decimal | None]:
    """The model's reading with every number in the KIND the user's words give it (entry 138; see the module text): returns (the form, what was corrected, one sentence each, the number that is left to ask about or None). Raises
    ValueError for a price limit the model made up and for more than 4 decimals in a limit, and `NeedsClarification` for more than one limit or more than one number to ask about. Never guesses the other way: a number is made
    shares only when it was written as a number of shares right before the stock, dollars only when it was written as money."""
    raw, notes = dict(raw), []
    orders = [{k: v for k, v in o.items() if k not in ("max_price", "min_price")} for o in raw.get("orders") or []]      # a limit and a floor are the request's, read from the words: one the model wrote on an order is not taken
    written = written_amounts(typed)
    money = {w.value for w in written if w.kind == "money"}
    # ---- the price limit: code reads it from the words, and wins
    limits = list(dict.fromkeys(w.value for w in written if w.kind == "ceiling"))
    model_limit = raw.get("max_price")
    if model_limit is not None:
        try:
            digits(model_limit)      # a number (a bool and a string are refused by the door)
        except ValueError:
            raise ValueError("max_price must be a number, or null when the person gave no price limit") from None
    if len(limits) > 1:
        raise NeedsClarification("you wrote more than one price limit (" + ", ".join(f"${canonical(v):,f}" for v in limits) + "): one limit for the whole request is supported. Send one limit, or one request for each stock.")
    if limits:
        limit = limits[0]
        if limit.as_tuple().exponent < -4:
            raise ValueError(f"a price limit has at most 4 decimals: ${canonical(limit):f} has more")
        if model_limit is None or digits(model_limit) != limit:
            notes.append(f"${canonical(limit):,f} a share is your price limit (the most one share may cost), not an amount to spend")
        raw["max_price"] = limit
    elif model_limit is not None:
        raise ValueError(f"max_price {model_limit} is not a price limit that is in what you typed: a price limit is a $ amount tied to one share ('under $50 a share'); leave max_price null when there is none")
    else:
        raw.pop("max_price", None)
    # ---- the price floor (entry 140): the least one share may cost; code reads it from the words, and wins, exactly as it does the limit
    floors = list(dict.fromkeys(w.value for w in written if w.kind == "floor"))
    model_floor = raw.get("min_price")
    if model_floor is not None:
        try:
            digits(model_floor)
        except ValueError:
            raise ValueError("min_price must be a number, or null when the person gave no minimum price") from None
    if len(floors) > 1:
        raise NeedsClarification("you wrote more than one minimum price (" + ", ".join(f"${canonical(v):,f}" for v in floors) + "): one minimum price for the whole request is supported. Send one, or one request for each stock.")
    if floors:
        floor = floors[0]
        if floor.as_tuple().exponent < -4:
            raise ValueError(f"a minimum price has at most 4 decimals: ${canonical(floor):f} has more")
        if limits and floor > limits[0]:
            raise NeedsClarification(f"your minimum price (${canonical(floor):,f} a share) is above your price limit (${canonical(limits[0]):,f} a share), so no price could pass both: say again what you want")
        if len({o.get("side") for o in orders}) > 1:
            raise NeedsClarification(f"a minimum price (${canonical(floor):,f} a share) goes with ONE trade, and you named a purchase and a sale: send the purchase and the sale as two requests, each with its own price")
        if model_floor is None or digits(model_floor) != floor:
            notes.append(f"${canonical(floor):,f} a share is your minimum price (the least one share may cost), not an amount to spend")
        raw["min_price"] = floor
    elif model_floor is not None:
        raise ValueError(f"min_price {model_floor} is not a minimum price that is in what you typed: a minimum price is a $ amount tied to one share ('over $20 a share', 'at $600 or more'); leave min_price null when there is none")
    else:
        raw.pop("min_price", None)
    # ---- a number after a money verb is shares or dollars: it is not read either way, the user is asked
    unclear = None
    spans = [(v, at) for v, at in ambiguous_numbers(typed) if v not in money]
    if spans:
        if len({v for v, _ in spans}) > 1:
            raise NeedsClarification("more than one amount of yours has no $ and no word dollars after a word like spend or invest (" + ", ".join(f"{canonical(v):,f}" for v in dict.fromkeys(v for v, _ in spans))
                                     + "): write each as dollars ('$1,000' or '1000 dollars') or as shares ('1000 shares of MU')")
        unclear = spans[0][0]
        for o in orders:
            for key in ("shares", "dollars"):
                if o.get(key) is not None and digits(o[key]) == unclear:
                    o[key] = None
        if raw.get("budget") is not None and digits(raw["budget"]) == unclear:
            raw["budget"] = None
    # ---- the kind of a number of a direct order: the code's kind wins
    for o in orders:
        name = o["symbol"].strip().lstrip("$") if "symbol" in o else " ".join(o["company"].split())
        as_shares = {n for n, _ in _order_phrases(typed, name, ticker="symbol" in o)}
        shares, dollars = o.get("shares"), o.get("dollars")
        if dollars is not None and digits(dollars) not in money and digits(dollars) in as_shares:
            o["shares"], o["dollars"] = dollars, None
            notes.append(f"{canonical(digits(dollars)):,f} before {name} has no $ and no word dollars, so it is {canonical(digits(dollars)):,f} shares, not dollars")
        elif dollars is not None and (digits(dollars) in limits or digits(dollars) in floors) and digits(dollars) not in money:
            o["dollars"] = None
            notes.append(f"{canonical(digits(dollars)):,f} is a price per share (your limit), not the dollars to spend on {name}")
        elif shares is not None and digits(shares) in money and digits(shares) not in as_shares:
            o["dollars"], o["shares"] = shares, None
            notes.append(f"${canonical(digits(shares)):,f} is money (it has a $ or the word dollars), so it is dollars, not shares of {name}")
    # ---- the kind of the budget and the count of a request that names ONE stock: "buy 10 of MU" is ten shares, not a budget of $10 and not the top 10
    if orders:
        raw["orders"] = orders
    budget, count = raw.get("budget"), raw.get("count")
    if budget is not None and digits(budget) not in money and (digits(budget) in limits or digits(budget) in floors):
        raw["budget"] = budget = None
        notes.append("a price per share (your limit) is not the money to spend")
    stocks = _named(raw, orders)
    if len(stocks) == 1 and raw.get("action") in ("buy", "order") and not asks_for_recommendation(typed):
        name, is_ticker = stocks[0]
        as_shares = {n for n, _ in _order_phrases(typed, name, ticker=is_ticker)}
        for key, number in (("budget", budget), ("count", count)):
            if number is None or isinstance(number, bool) or digits(number) not in as_shares or digits(number) in money or (key == "count" and digits(number) in count_written(typed)):
                continue
            if orders and (orders[0].get("shares") is not None or orders[0].get("dollars") is not None):
                continue
            order = orders[0] if orders else {"side": "buy", ("symbol" if is_ticker else "company"): name}
            order["shares"], order["dollars"] = number, None
            raw = {**raw, "action": "order", key: None, "orders": [order], "symbols": [], "companies": []}
            notes.append(f"{canonical(digits(number)):,f} before {name} has no $ and no word dollars, so it is {canonical(digits(number)):,f} shares of {name}, not " + ("a budget" if key == "budget" else "a number of companies"))
            break
    if raw.get("action") in ("buy", "research") and asks_for_recommendation(typed):      # a number of shares TYPED WITH a recommendation (entry 140, C1): it is the shares of the stock it stands before, or of every stock; the model may have put it in the budget or the count
        every = shares_each_written(typed)
        as_shares = {n for name, is_ticker in stocks for n, _ in _order_phrases(typed, name, ticker=is_ticker)} | set(every)
        counts = count_written(typed)
        for key in ("budget", "count"):
            number = raw.get(key)
            if number is None or isinstance(number, bool) or digits(number) in money or digits(number) not in as_shares or (key == "count" and digits(number) in counts):
                continue
            raw[key] = None
            notes.append(f"{canonical(digits(number)):,f} is a number of shares (no $ and no word dollars), not " + ("a budget" if key == "budget" else "a number of companies"))
        if every and raw.get("each") and raw.get("budget") is None:
            raw["each"] = False      # "5 shares of each": the number is the shares of every stock, not an amount of money for each
    return raw, notes, unclear


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
    known = {"action", "topic", "symbols", "companies", "count", "allocation", "budget", "each", "max_price", "min_price", "raise_funds", "orders", "news_days"}
    if set(raw) - known:
        raise ValueError(f"unknown fields: {', '.join(sorted(map(str, set(raw) - known)))}")
    orders = raw.get("orders", [])
    if not isinstance(orders, list) or not all(isinstance(o, dict) and {"side"} <= set(o) <= {"side", "symbol", "company", "shares", "dollars", "max_price", "min_price"} and ("symbol" in o) != ("company" in o)
                                                and all(isinstance(o[k], str) for k in ("symbol", "company") if k in o)
                                                and all(o.get(k) is None or (isinstance(o[k], (int, float)) and not isinstance(o[k], bool)) for k in ("shares", "dollars", "max_price", "min_price")) for o in orders):
        raise ValueError('orders must be a list of {"side": "buy" or "sell", "symbol": a ticker in capitals, or "company": a company name, "shares": a number (a decimal such as 0.5 is allowed, with at most 6 decimals), or null when the person wrote no number of shares, "dollars": a number, or null when the person wrote no amount of money for that stock}')
    for number, o in enumerate(orders):
        if "symbol" in o:
            try:
                clean = check_symbol(o["symbol"].strip().lstrip("$"))
            except ValueError:
                clean = None
            if clean != o["symbol"].strip().lstrip("$"):
                raise ValueError(f"orders[{number}]: {o['symbol']!r} is not a ticker in capitals: a company's name goes in company, not in symbol")
    model_shares: dict = {}
    if raw.get("action") == "order" and asks_for_recommendation(typed):      # the words ask for a recommendation (entry 98): not a direct order, whatever the model said; the stocks it listed are the ones to be recommended
        dollars = [o["dollars"] for o in orders if o.get("dollars") is not None]
        if len(dollars) > 1:
            raise NeedsClarification("you asked for a recommendation and gave an amount for each stock: a recommendation is bought with ONE total. Give one total, or order the stocks yourself with no recommendation.")
        raw = {**raw, "action": "buy", "orders": [], "symbols": [o["symbol"].strip().lstrip("$") for o in orders if "symbol" in o], "companies": [" ".join(o["company"].split()) for o in orders if "company" in o],
               **({"budget": dollars[0]} if dollars and raw.get("budget") is None else {})}      # the amount of the order is the money of the request: it is never dropped (the Account Manager would otherwise spend all the cash)
        model_shares = {o.get("symbol") or o.get("company"): o["shares"] for o in orders if o.get("shares") is not None}      # the model's own reading of them; the shares of the request are read from the words (`typed_shares`)
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
    if raw.get("each") is None:      # a model that writes null for each means false
        raw.pop("each", None)
    elif not isinstance(raw["each"], bool):
        raise ValueError("each must be true or false")
    if raw.get("action") == "news":      # entry 149: the user only wants to see the news: no number of shares, no price and no permission to sell is read from the words
        return _news_intent(raw, typed)
    raw.pop("news_days", None)      # a window means nothing to a request that is not for the news
    said = raw.pop("raise_funds", None)      # the permission to sell holdings (entry 142, R2): the words decide, the model only has to agree
    if said is not None and not isinstance(said, bool):
        raise ValueError("raise_funds must be true or false")
    if said and not raise_funds_in(typed):
        raise ValueError("raise_funds is true, but you did not say to raise funds: it is true only when the person said so ('raise funds if needed', 'sell some stock to pay for it'); leave it false otherwise")
    if raise_funds_in(typed) and raw.get("action") in ("buy", "order"):
        raw["raise_funds"] = True
    raw, corrected, unclear = _read_numbers(raw, typed)      # entry 138: every number in the kind the user's words give it, a price limit from the words, a number after "spend" left to ask about
    orders = raw.get("orders", [])
    if raw.get("action") == "order" and (raw.get("max_price") is not None or raw.get("min_price") is not None):
        raw = {**raw, "orders": [{**o, **{k: raw[k] for k in ("max_price", "min_price") if raw.get(k) is not None}} for o in orders]}      # the request's price limit and price floor are every order's
    if raw.get("action") in ("buy", "research") and not raw.get("symbols") and not raw.get("companies") and not orders:      # a phrase in quotes is the theme, word for word (entry 113)
        phrase = quoted_theme(raw.get("topic"), typed)
        if phrase is not None:
            raw = {**raw, "topic": phrase}
    try:
        intent = TradeIntent(request=typed, corrected=list(corrected), unclear_number=unclear, **raw)
    except ValidationError as error:
        first = error.errors()[0]
        raise ValueError(f"{'.'.join(str(x) for x in first['loc']) or 'the form'}: {first['msg']}") from None
    except ValueError as error:   # check_symbol
        raise ValueError(str(error)) from None
    if intent.action in ("buy", "research") and skips_recommendations(typed) and as_direct_order(intent, typed) is None:      # S3: "skip recommendations" and no stock to buy: only the recommendations choose stocks for a theme
        raise NeedsClarification(SKIP_NEEDS_STOCKS)
    # every field must be in the user's own words
    numbers = _numbers(typed)
    for order in intent.orders:
        check_order(order, typed, intent.each)
        check_dollars(order, typed)      # entry 122: the dollars of an order are an amount the user wrote as money
    check_each(intent, typed)      # entry 125: ONE amount for every stock only when the user said each (a word that says it, the same amount on every order or as the budget)
    check_dollars_each(intent, typed)      # ... and an amount is the dollars of as many orders as it was written for (once for several stocks: one total, never each, unless each was said)
    check_budget(intent, typed)      # entry 138: the budget is an amount the user wrote as money, whatever the action
    if intent.action == "order":
        check_amount_was_read(intent, typed)      # entry 122: an amount of money in the words and none in the reading is refused, never turned into all the cash
        return intent
    check_named(intent, typed)
    if intent.count is not None and Decimal(intent.count) not in numbers:
        raise ValueError(f"the number {intent.count} is not in what you typed")
    if intent.allocation == "even" and not EVEN_CUE.search(typed):
        raise ValueError("an even split was not asked for in what you typed")
    if intent.action == "buy" and not BUY_CUE.search(typed):
        if not asks_for_recommendation(typed):
            raise ValueError("buying was not asked for in what you typed")
        intent = TradeIntent.model_validate({**intent.model_dump(), "action": "research", "each": False, "raise_funds": False, "corrected": [*intent.corrected, "you asked for recommendations and said nothing about buying: this is a look, nothing will be bought"]})      # a program decides it from the user's words (the model read "Recommend 3 oil stocks" as a purchase)
    direct = as_direct_order(intent, typed)
    if direct is not None:
        for order in direct.orders:
            check_dollars(order, typed)      # the model read "buy" with the stocks named and an amount for each (entry 125): the orders are made here, and the checks of an order's dollars apply to them
        check_dollars_each(direct, typed)
        check_amount_was_read(direct, typed)
        return direct
    if intent.action in ("buy", "research") and intent.count is None and not intent.symbols and not intent.companies and one_stock_asked(typed):      # "the most recommended shoe company": ONE company (entry 146), never the default of 3
        intent = intent.model_copy(update={"count": 1, "corrected": [*intent.corrected, "you asked for one company (the most recommended, the best, the top one): only the first is recommended" + (" and bought" if intent.action == "buy" else "")]})
    if intent.action == "buy":      # a number of shares typed WITH a recommendation (entry 140, C1): read from the words, it is each approved stock's own limit (C2)
        of, every = typed_shares(intent, typed)
        gone = [f"{sh.text(sh.quantity(n))} shares of {name} is not a number of shares you typed, so it is not used" for name, n in model_shares.items() if name not in of and not every]
        loose = sorted(shares_written(typed) - set(of.values()) - ({every} if every is not None else set()))      # a number written as SHARES that no stock and no "each" took (entry 146): it is NEVER dropped
        if loose:
            if len(loose) == 1 and intent.count == 1 and not of and loose[0] == loose[0].to_integral_value():
                every = loose[0]      # "buy 5 shares of the most recommended shoe company": the 5 shares of the ONE company
            else:
                raise NeedsClarification(f"you wrote {' and '.join(sh.text(n) for n in loose)} shares but not which stock they are for: say how many companies and the shares of each ('buy 5 shares of each of the top 3 shoe stocks'), "
                                         "or ask for one ('buy 5 shares of the most recommended shoe company'). Nothing was bought: a number of shares is never dropped, and the money is never spent in its place.")
        if of or every is not None or gone:
            intent = intent.model_copy(update={"shares_of": of, "shares_each": every, "corrected": [*intent.corrected, *gone]})
    return intent


def as_direct_order(intent: TradeIntent, typed: str) -> TradeIntent | None:
    """A request to BUY stocks the user NAMED, with nothing that asks for a recommendation, is a direct order (design choices entry 98: a recommendation is not needed for a direct order unless the user asks for one).
    The model sometimes reads such a request as "buy" (a list of tickers and the choice left to the desk) and sometimes as "order"; this is the program's check of the user's own words, so the answer does not depend on
    the model: named tickers or companies, no theme, no count ("the best 2 of ..."), no word that asks for a recommendation, and the user's own buying word before each stock. Returns the direct order, or None when the
    request is not one (it stays what the model said)."""
    if intent.action != "buy" or intent.topic or intent.count is not None or not (intent.symbols or intent.companies) or asks_for_recommendation(typed):
        return None
    try:
        each = intent.each_dollars      # "Buy $3,000 each of XOM and MSFT" read as a purchase of named stocks with the amount for each as the budget (entry 125): the amount is the dollars of every order
        limit, floor = intent.max_price, intent.min_price      # a price limit (entry 138) and a price floor (entry 140) are every order's
        orders = [DirectOrder(side="buy", symbol=symbol, dollars=each, max_price=limit, min_price=floor) for symbol in intent.symbols] + [DirectOrder(side="buy", company=name, dollars=each, max_price=limit, min_price=floor) for name in intent.companies]
        converted = TradeIntent(request=intent.request, action="order", orders=orders, budget=intent.total_budget, allocation=intent.allocation, each=intent.each, max_price=limit, min_price=floor, raise_funds=intent.raise_funds, corrected=intent.corrected, unclear_number=intent.unclear_number)
        for order in converted.orders:
            check_order(order, typed, intent.each)
    except ValueError:      # includes a pydantic ValidationError
        return None
    return converted


# ---------------------------------------------------------------- the number that is shares or dollars (entry 138, D5)

SHARES_WORD = re.compile(r"\bshares?\b|^\s*s\s*$", re.I)
DOLLARS_WORD = re.compile(r"\$|\bdollars?\b|\bbucks?\b|\busd\b|\bmoney\b|^\s*d\s*$", re.I)


def read_unit_answer(answer, number: Decimal) -> Literal["shares", "dollars"] | None:
    """The unit the user's answer to "shares or dollars?" names: `shares` (the word shares), `dollars` (a `$`, the word dollars, bucks, usd or money), or None when it names neither, both, or another number than the one asked about.
    A program reads it, not the model: it decides what real money is spent on."""
    text = answer if isinstance(answer, str) else ""
    shares, dollars = bool(SHARES_WORD.search(text)), bool(DOLLARS_WORD.search(text))
    if shares == dollars or any(n != number for n in _numbers(text)):
        return None
    return "shares" if shares else "dollars"


def settle_unit(intent: TradeIntent, unit: Literal["shares", "dollars"]) -> TradeIntent:
    """The intent with its `unclear_number` settled as `unit` (the user's answer): dollars become the dollars of the one order named, else the budget; shares become the shares of the one order named. Raises ValueError, with
    the reason, when the number cannot be that (shares need ONE stock named; a request to be recommended has no shares), and for a number that is not an amount of money to the cent."""
    number = intent.unclear_number
    if number is None:
        return intent
    data = intent.model_dump()
    data["unclear_number"] = None
    orders = data["orders"]
    if unit == "shares":
        if intent.action != "order" or len(orders) != 1 or orders[0].get("shares") is not None or orders[0].get("dollars") is not None:
            raise ValueError(f"{canonical(number):,f} shares: a number of shares goes with ONE stock you named. Say how many shares of each stock, or give the dollars ('$1,000')")
        orders[0]["shares"] = number
    elif intent.action == "order" and len(orders) == 1 and orders[0].get("shares") is None and orders[0].get("dollars") is None:
        orders[0]["dollars"] = number
    else:
        data["budget"] = number      # one amount for several stocks, or for the companies to be recommended: ONE total
    data["corrected"] = [*data.get("corrected", []), f"{canonical(number):,f} was settled with you as {unit}"]
    try:
        return TradeIntent.model_validate(data)
    except ValidationError as error:
        first = error.errors()[0]
        raise ValueError(f"{'.'.join(str(x) for x in first['loc']) or 'the form'}: {first['msg']}") from None
