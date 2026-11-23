"""The Account Manager, as an LLM agent: RaiseFunds, a Tree of Thought for "what do I sell to buy this?" (design choices entry 42).

    am = AccountManager()                                           # the LLM of the `account_manager` role (ACCOUNT_MANAGER_MODEL, openai/gpt-oss-120b by default)
    result = am.raise_funds(load_portfolio("configs/portfolios/jacob_pyrett.json"), "MU", 2)
    print(report(result))

    python -m prometheus.agents.account_manager.account_manager --buy MU --shares 2          # LIVE: Finnhub quotes + OpenRouter calls (paid)

The pattern is the Advisor's (prometheus/agents/advisor/advisor.py): a class that takes a chat model (a scripted one in tests) or builds the role's model with `get_chat_model`, one
public method that does the job, a prompt module, a JSON reply that is parsed defensively with one retry, and CODE that checks everything the model says. Here the model is the thought
generator of a tree search and the checks, the scores, the pruning and the final choice are the program's (prometheus/reasoning/raise_funds.py).

What `raise_funds` does, in order:
  1. asks the DATA CLERK for the current price of the target and of every holding: a work order with `action_type="fetch_quote"`, answered by `DataClerk.request_quotes` (the
     Data Clerk's `get_quote` tool, one Finnhub call per symbol). The Account Manager never reads a price from anywhere else, and a missing quote ends the request.
  2. builds the problem (the holdings, what is sellable under T+1, the cost of the purchase) and runs the tree search, calling the model once per node it expands.
  3. returns the result as one dict: the chosen plan, the alternatives, the reasoning trace, the quotes it used, and what the model run cost.

This module is also the "account_manager_funds" node of the desk graph (prometheus/langgraph/desk.py, `build_desk(with_funds=True)`).

The same class is the user's account book (design choices entry 72). With `store` and `account_id` it reads and writes the Postgres account: add_funds, list_portfolio, set_risk_level,
account_balance, the realized gain, the trade count, the approved and denied recommendation counts, and `users` for the name, the account number and the fields a later screen will show.
That path does not call a model or a quote. The desk still uses the JSON profile and portfolio unless it is given the store.

The Account Manager is also the Trader's only door (Trader v1, design/trader_v1_final_spec.md): `connect` attaches the account book after construction, `trading_portfolio` is the portfolio the Trader works from,
`request_prices` and `request_bars` are the quotes (paper) and the stored daily bars (backtest) it is given, and `perform_trades` hands a decided order to the Trader. THE FUNDS ARE THE ACCOUNT MANAGER'S ALONE
(design choices entry 81, `handshake.py`): the Trader holds no cash and no portfolio, it asks `approve_trade` for each trade (the Account Manager reads the funds and holdings as they are now and says for how many shares)
and reports each fill AT ONCE with `record_fill` (applied to the portfolio and, with a saved account, booked in ONE Postgres transaction), and only then asks for the next trade, so it cannot overdraw the funds.
`apply_earlier_fills` and `current_portfolio` are the two small reads that belong to it. The Trader never asks the Data Clerk, the portfolio or the database itself. `notify_trades` stays for a caller that reports a
list of fills to a portfolio it holds.

A NEUTRAL RECOMMENDATION IS AN APPROVAL (design choices entries 91 and 107): with `ALLOW_NEUTRAL_RECOMMENDATIONS=true` every company the Advisor scored neutral, whoever named it, counts as approved when `decide_trades` works with the
work order's recommendation result, up to the number of companies the request asked for (`count`), and the Trader may buy it (`neutral.py`); it is sized together with the approved purchases, in one call.

IGNORE CRITIC (design choices entry 148): when the user ticked the box for a run the desk skips the Critic's node and every recommendation of the Advisor counts as approved (`ignore_critic.py`, `origin: "advisor"`), decided like a neutral one and sized
with the neutral ones in the same call; with the box off nothing here changes.

THE ACCOUNT MANAGER DOES NO RISK MANAGEMENT (Jacob, 2026-11-15): it delegates tasks to the other agents, updates the user's profile and portfolio, and works with verifying the trades the Trader has made. The only check on
an order is that the funds cover it (entry 91).

A DIRECT ORDER (design choices entry 87): when the user names the trade ("Buy 50 shares of MSFT") `understand_request` reads it as `orders`, the desk skips the Data Clerk's search, the Advisor and the Critic, and `decide_direct` prices
each order with the Data Clerk and decides it (the user's own order is the approval), and the Trader places it through the same handshake: as typed or not at all.

A DIRECT ORDER WHOSE LIMIT IS NOT MET WAITS (design choices entry 140; `orders.py`, design/limit_orders_spec.md Part B): a purchase with a ceiling ("buy 10 MU at $90 or less") whose quote is above it, and a sale with a floor ("sell 10 MSFT at $600 or more")
whose quote is below it, become an OPEN ORDER, a DAY order on the paper broker with its cash (a purchase) or its shares (a sale) SET ASIDE: `decide_direct` places it in the account book (`_place_waiting`: one transaction under the account's row locks), what is
set aside is not available to anything else (the book's one door: `list_portfolio` returns the AVAILABLE cash as `cash_avail` and the holding with the reservations beside it, `available_cash`, `open_orders`, `cancel_open_order`), and the web server's watcher
fills it through the SAME funnel when the quote reaches the limit (`fill_open_order`: the order's own hold is available to its own trade and to nothing else, `_portfolio`; the booking marks the order filled in the same transaction) or expires it at the close.

The Account Manager decides HOW MUCH to buy or sell (design choices entry 80: the Advisor and the Critic only deal with recommendations): `decide_trades` sizes every purchase the Critic approved from the user's money
(`sizing.py`; the only check is that the funds cover the order; Auto mode spends the funds round-robin, a share of each stock in turn; Auto mode asks nothing, and a recommendation the Critic denied is returned to the user: entry 104). A sale is never a recommendation (entry 105): it is the user's own direct order or RaiseFunds. It asks the Data Clerk for the current prices of the holdings first, when it needs them.

DECIMAL MONEY (design choices entry 112; Jacob, 2026-11-16: *"Need to always use decimal.Decimal type when dealing with money and share counts"*): every dollar amount, every price a trade is sized or filled at and every share count that this class
holds, passes or returns is a `Decimal`. A quote, a deal price, a stored bar or the user's own words come in as the int or float that `json` or an API client gave and are turned into a Decimal ONCE, at the door (`_prices_of`: `money.price`, four places,
half up); after that there is no float, and a float handed to the handshake is a TypeError that stops a test instead of putting a wrong cent in the account.

    python -m prometheus.agents.account_manager.account_manager --account avery_chen
"""
import argparse
import dataclasses
import json
import math
import re
import sys
import threading
import time
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from prometheus.agents.account_manager import funding, handshake, ignore_critic, neutral, orders, sizing
from prometheus.agents.critic import feedback
from prometheus.agents.account_manager.config import load_config as load_account_manager_config
from prometheus.agents.account_manager.intent import NeedsClarification, TradeIntent, check_typed, parse_intent, read_unit_answer, settle_unit
from prometheus.common import shares as sh
from prometheus.common.money import canonical, digits, money as money_door, price as price_door
from prometheus.common.market_calendar import eastern_date
from prometheus.common.portfolio import Portfolio, after_purchase, apply_fill, load_portfolio, trade_amount
from prometheus.common.trading_clock import trading_now
from prometheus.common.user_account import UserDirectory
from prometheus.common.user_profile import RiskView, UserProfile, load_profile
from prometheus.common.work_order import DIRECT, MAX_QUOTE_SYMBOLS, DataResult, WorkOrder, decision_dollars, decision_max_price, decision_min_price, decision_of, decision_pot, iso_date, new_work_order, verify_trade
from prometheus.models.chat import get_chat_model, is_rate_limited
from prometheus.models.chat import model_name as chat_model_name
from prometheus.prompts import account_manager_request as request_prompt
from prometheus.prompts.account_manager import RETRY, system_prompt, user_message
from prometheus.rag.loaders import check_symbol
from prometheus.reasoning import raise_funds as rf
from prometheus.common import langsmith_tracing
from prometheus.common.trace import traced
from prometheus.security.guard import Guard

DEFAULT_PORTFOLIO = Path(__file__).resolve().parents[3] / "configs" / "portfolios" / "jacob_pyrett.json"
DEFAULT_PROFILE = Path(__file__).resolve().parents[3] / "configs" / "profiles" / "jacob_pyrett.json"
REQUESTER = "account_manager"


def parse_thoughts(text: str) -> list:
    """The `thoughts` list from the model's reply. Raises ValueError, with a short reason, if there is none."""
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found")
    try:
        data = json.loads(body[start:end + 1])
    except json.JSONDecodeError:
        raise ValueError("not valid JSON") from None
    if isinstance(data, dict) and isinstance(data.get("plans"), list):   # whole plans (entry 50)
        return data["plans"]
    if not isinstance(data, dict) or not isinstance(data.get("thoughts"), list):
        raise ValueError("no thoughts list")
    return data["thoughts"]


def top_recommendation(order) -> str | None:
    """The symbol to buy from a work order the Advisor has answered: the recommendation with the highest CALCULATED score, if that score is above 50 (more verified bullish signals than
    bearish ones). None when there is none. Code, not a model: the ranking of the design ("recommendations are ranked in order of the final score")."""
    ranked = [r for r in order.recommendations if r.get("supported") and isinstance(r.get("score"), (int, float)) and r["score"] > 50]
    return max(ranked, key=lambda r: r["score"])["symbol"] if ranked else None


PREFERRED_PROVIDERS = ("cerebras", "groq")   # the fast hosts of gpt-oss-120b on OpenRouter (entry 50)


class AccountManager:
    def __init__(self, model=None, model_id: str | None = None, clerk=None, effort: str | None = "auto", only_green: bool = True, never_liquidate: bool = True, green_first: bool = True,
                 explore: bool = True, lean: bool = True, whole_plans: bool = False, parallel: bool = True, up_to: bool = True, deadline: float | None = 90.0,
                 providers: tuple[str, ...] | str | None = "auto", profile: UserProfile | str | Path | None = None, store=None, account_id: int | None = None, screen=None,
                 fractional_even_split: bool | None = None, guard: Guard | None = None):
        """model: a chat model (anything with `invoke`; a scripted one in tests); otherwise OpenRouter through `get_chat_model` with the model of the `account_manager` role
        (ACCOUNT_MANAGER_MODEL), or `model_id`. clerk: the Data Clerk that is asked for prices (anything with `request_quotes`; a fake in tests); built when first needed.
        effort (entry 46): "low", "medium" or "high", OpenRouter's `reasoning` effort for a model that thinks before it answers (gpt-oss-120b does): a thought is three short JSON steps and most of a
        call's time is the model thinking, so "low" shortens every call. None leaves the model's own default; "auto" (the default) is "low" for gpt-oss and None for any other model (it made
        qwen3-30b slower and worse, entry 46); higher efforts only cost time (entry 50).
        only_green and never_liquidate (entry 47): the rules of the evaluator, ON unless you turn them off. `only_green`: only a holding whose price is above its cost basis may be sold, so no sale takes
a loss. `never_liquidate`: at least one share of every holding is kept, so no position is sold out. They are the defaults of every method below; a method's own argument overrides them.

        THE DEFAULTS ARE THE WINNING SETTINGS (entry 50), all on: `green_first` (the score prefers the most profitable holdings), `explore` (the beam ranks by score and progress, and a thought of
        one kind is asked again), `lean` (no sell of more shares than needed), `parallel` (the plans of a level call the model at the same time), `up_to` (a number of shares means up to that many)
        and `deadline` (90 seconds a purchase, then the best plan so far; None for no limit). NOT a default: `whole_plans` (a thought is a whole plan, one to three rounds of the model): measured
        slower, 121 s against 33 to 55 s for the same purchase (entry 50). `AccountManager.classic()` turns all of them off: the behaviour of entries 42 to 47, which the older tests use.

        providers (entry 50): the OpenRouter providers to prefer, in order (the others are the fallback). "auto" (the default) is Cerebras then Groq for gpt-oss and OpenRouter's own choice for any other
        model; None is OpenRouter's own choice. Measured: the same gpt-oss-120b answers at about 55 tokens a second by default, 650 on Cerebras and 370 on Groq: the largest single gain in time.
        The model is the same; only where it runs changes (the provider sees the prompt: prices, holdings and quantities, no keys).

        profile (Critic v1, design choices entry 56): the user's `UserProfile` (risk tolerance, Ask or Auto mode), or the path of its JSON file; `configs/profiles/jacob_pyrett.json` when left
        out, read when first needed. When `account_id` is set and no profile was given, the profile is the account row instead. The Account Manager is the only agent that holds it: the Critic is given
        `risk_view()` (no mode, no money), and `decide_trades` turns the Critic's verdicts into permissions to trade and sizes them. None of that calls the model (`AccountManager(model=object())` is enough for it).

        store and account_id (entry 72): the Postgres account book, both or neither (or attached later with `connect`, as the desk does for `--account`). The book methods below refuse to run without them. `decide_trades` writes
        each verdict when they are set, and the Trader's fills are booked one at a time through `record_fill` when the run asks to `book`; `trading_portfolio` is then the saved account, not a file.

        screen (entry 82): where the Account Manager speaks to the user, a callable that takes one line of text, or None. The user interacts with the Account Manager only: when a trade fails (the book refused it, or it
        could not be confirmed) it shows the error and the account as it stands through `tell_user` and `show_portfolio`, at that moment.

        fractional_even_split (entry 110): whether "split it evenly" buys equal DOLLARS of each stock by buying a fraction of a share. None (the default) is the switch `sizing.fractional_even_split` of `configs/account_manager_v1.json`;
        True or False overrides it for this manager (False is the whole-share split of entry 108, which the tests of that entry use). Either way a connected account book that cannot hold a fraction (`AccountStore.holds_fractions`)
        gets the whole-share split, and the user is told why.

        guard (entry 116): the security layer's door for what the Account Manager says to the user (`tell_user`): each line is screened and a secret or another account's number in it is redacted in place. None is `Guard.code_only()`, made when first
        needed: the redaction is code and needs no model."""
        self._check_book(store, account_id)
        self.fractional_even_split = fractional_even_split
        is_oss = "gpt-oss" in (model_id or chat_model_name("account_manager"))
        if effort == "auto":
            effort = "low" if is_oss else None
        if providers == "auto":
            providers = PREFERRED_PROVIDERS if is_oss else None
        extra = {**({"reasoning": {"effort": effort}} if effort else {}), **({"provider": {"order": list(providers), "allow_fallbacks": True}} if providers else {})}
        self.model = model or get_chat_model("account_manager", model=model_id, **({"extra_body": extra} if extra else {}))
        self._clerk = clerk
        self._profile = profile
        self._profile_from_file = False  # True once `profile` has read a file that nobody gave (the default JSON file, or a path not read until then): `connect` forgets such a profile
        self.store = store
        self.account_id = account_id
        self._guard = guard
        self.screen = screen                       # the user's screen: a callable that takes one line of text (the session gives its own `out`). The user talks to the Account Manager and to nobody else, so what the Trader
                                                   # cannot say to the user (a trade that failed) the Account Manager says (`tell_user`, `show_portfolio`); None says nothing (the same text is on the order and in the reports)
        self.only_green, self.never_liquidate = only_green, never_liquidate
        self.green_first, self.explore, self.lean, self.whole_plans, self.parallel, self.up_to, self.deadline = green_first, explore, lean, whole_plans, parallel, up_to, deadline
        self._lock = threading.Lock()  # the plans of a level may call the model at the same time (parallel): the counters of the run are shared
        self.run: dict = {}
        self.intent_run: dict = {}   # what the last `understand_request` cost: the model, the calls, the retries and the tokens
        self.last_sizing: dict = {}                # what the last `clarify_sizing` asked and was answered, and what it cost (design choices entry 85)
        self.last_missing: dict = {}               # what the last `name_orders` asked about the stocks the Data Clerk did not find (Ask mode): the cards, who worded each question and why, and why the order stopped (entry 102)
        self._session = False                      # a trading run is open (`trading_portfolio` was called): the handshake with the Trader works on it (design choices entry 81)
        self._working: Portfolio | None = None     # the Account Manager's WORKING portfolio of that run: what `approve_trade` reads and `record_fill` changes when nothing is booked. Never the Trader's
        self._booking = False                      # the run books each fill in the Postgres account book (asked for with `book`, and a saved account is connected)
        self._book_unavailable = False             # booking was asked for and no saved account is connected: said once, in the first Booking
        self._run_fills: dict[str, dict] = {}      # the fills `record_fill` was told in the open run, by trade id (a copy of each filled trade): what the run has spent of its pot besides what the work order already shows (entry 111, M1)
        self._filling = None                       # the waiting order (an `OpenOrder`) this manager is FILLING right now (`fill_open_order`, entry 140): its own hold is available to ITS trade only, and its booking marks it filled

    @classmethod
    def classic(cls, model=None, clerk=None, **options):
        """An Account Manager with every default of entries 45 to 50 turned off (the rules of entries 42 to 44: a loss may be taken, a position may be sold out, one sell per thought, exact
        numbers of shares, no limit, one call after the other). The tests of the earlier entries use it; `options` override any argument."""
        return cls(model=model, clerk=clerk, **{"effort": None, "only_green": False, "never_liquidate": False, "green_first": False, "explore": False, "lean": False, "whole_plans": False,
                                                 "parallel": False, "up_to": False, "deadline": None, "providers": None, **options})

    @property
    def profile(self) -> UserProfile:
        """The user's profile, read when first needed: the account row when this manager is connected and no profile was given, otherwise the JSON file. Raises ValueError, naming the file, when a file is not a valid profile."""
        if not isinstance(self._profile, UserProfile):
            if self._profile is None and self.account_id is not None:
                self._profile = self.store.load_profile(self.account_id)
            else:
                self._profile = load_profile(self._profile or DEFAULT_PROFILE)
                self._profile_from_file = True
        return self._profile

    @profile.setter
    def profile(self, value: UserProfile | str | Path | None) -> None:
        self._profile = value
        self._profile_from_file = False  # a profile somebody set is a profile that was given

    def risk_view(self) -> RiskView:
        """What the Critic may know about the user: the risk tolerance, age, horizon, exclusions and minimum price. Never the name, the trading mode or any money (the user's money is the Account Manager's, which only checks that the funds cover an order)."""
        return self.profile.risk_view()

    def fractional_split(self, balance: str | None) -> bool:
        """Whether an EVEN split of this run is made in fractional shares (design choices entry 110): the switch says yes (`fractional_even_split`, else `sizing.fractional_even_split`) AND the account book can hold a fraction
        (no book is connected: a JSON portfolio holds any quantity; a connected book: `AccountStore.holds_fractions`). When the switch says yes and the book cannot, the split is in whole shares and the user is told so in one line
        (only when the split IS even: `balance` is "even"). Nothing is rounded into an integer column. A database that cannot be asked counts as one that cannot."""
        return self.fractions_on("the split is in whole shares" if balance == "even" else "")

    def fractions_on(self, say: str = "") -> bool:
        """Whether this run may buy a fraction of a share: the switch says yes (`fractional_even_split`, else `sizing.fractional_even_split`) AND the account book can hold one (`_book_holds_fractions`). When the switch says yes
        and the book cannot, `say` (when given) is told to the user in one line with the reason and the fix. Used for the even split (entry 110) and for a direct order with money (entry 111, M3)."""
        wanted = load_account_manager_config().sizing.fractional_even_split if self.fractional_even_split is None else self.fractional_even_split
        if not wanted:
            return False
        holds = self._book_holds_fractions()
        if not holds and say:
            self.tell_user(f"      {say}: the account book has no room for a fraction of a share yet (run `bash scripts/setup_postgres.sh` to change that)")
        return holds

    def _book_holds_fractions(self) -> bool:
        """Whether the account book can hold a fraction of a share: True with no book (a JSON portfolio holds any quantity), else `AccountStore.holds_fractions`; a database that cannot be asked counts as one that cannot."""
        if self.store is None or self.account_id is None:
            return True
        try:
            return bool(self.store.holds_fractions())
        except Exception:
            return False

    @traced("account_manager.decide_trades")
    def decide_trades(self, order: WorkOrder, portfolio: Portfolio, approver=None, now=None, mode: str | None = None, balance: str | None = None, budget: Decimal | None = None,
                      clarify=None, said_how_much: bool = False, request_text: str = "", count: int | None = None, each: Decimal | None = None, max_price: Decimal | None = None,
                      min_price: Decimal | None = None, shares_of: dict | None = None, shares_each: Decimal | None = None, raise_funds: bool = False, top_pick: bool = False) -> WorkOrder:
        """The Critic's verdicts become trade decisions (`order.trade_decisions`, one per review; see `decisions.py` for the rules), and the Account Manager decides HOW MUCH (design choices entry 80: the Advisor and the
        Critic only deal with recommendations): a denial is final, an approval that ran out is not traded, every approved purchase is sized from the user's money (`sizing.py`: the only check is that the funds cover the order), and an approved recommendation
        is traded in Auto mode, where NOTHING is asked (design choices entry 104: there is no second mark and a review that wants a person was already denied by the Critic); in Ask mode it goes to the user, through `approver`
        (called with the scorecard; True, False or None for no answer = `pending_user`). A DENIED recommendation is never traded and is returned to the user (`session.denied_lines`). `mode` overrides the profile's `trade_mode` for one run. `balance` says how the money is spread over the purchases ("even": equal dollars, "round_robin": a share of each in turn;
        left out, the money is spent round-robin in both modes, entry 106) and `budget` is the funds requested (dollars): the user's own words, "split it evenly" and "spend $5,000".

        HOW MUCH, ASKED OF THE USER (design choices entry 85). In Ask mode, when `clarify` (the channel to the user: a callable that takes a card and returns what they typed, or None) is given, an approved purchase is
        waiting to be sized and the request did not already say how much (`said_how_much` False and no `budget`), the Account Manager first asks the user, with a bounded ReAct loop (`clarify_sizing`): a percentage of the
        account, a dollar amount, a number of shares, or "as much as I can". What comes out sizes the purchases (`balance`, `budget`, and the shares asked for each company), under the cash; the user then approves each
        trade as before. Auto mode never asks, and without `clarify` nothing is asked. `request_text` is the user's request, shown to the model so that it does not ask what was said.

        The current prices of the symbols the portfolio HOLDS are asked of the Data Clerk first (one quote order: `request_prices`), for the equity ("20% of my account", and the record); a portfolio with no holding asks for nothing (and an
        order with no purchase to size asks for nothing), and a price that cannot be had is a message on the order and that holding counts at its cost basis. `decisions.decide_trades` itself calls no API.
        A NEUTRAL RECOMMENDATION IS AN APPROVAL, whoever named the company (entry 107): with `ALLOW_NEUTRAL_RECOMMENDATIONS` on, every company the Advisor scored neutral counts, and it is bought WITH the approved ones, in ONE sizing
        (`decisions.decide_trades`: "evenly among the 3" is three equal amounts, the default one round-robin over all). `count` is the number of companies the request asked for (the form's own `count`; None when the request gave
        none, and then nothing is cut): the Critic's approved recommendations come first, then the neutral companies, the Advisor's best score first, until it is reached; a denied recommendation does not count; a neutral company
        beyond it gets NO decision and the user is told it was neutral and not needed.

        `each` (design choices entry 125; "buy the top 3 oil stocks with $3,000 each") is the amount the user said for EACH purchase this run decides, the Critic's approved ones and the neutral ones alike: every purchase is limited to its own
        amount (the plan of the answer "an amount for each company", `clarify.plan_each_dollars`; whole shares, the leftover stays cash) and the POT of the run is the amount times the number of purchases that can be sized, never above the
        cash. Nothing is asked in either mode (the user already said how much), and no split is made (`balance` is dropped: every purchase has its own cap). It is one way to say how much, so it does not go with a `budget`.

        `max_price` (design choices entry 138; "the top 3 AI stocks under $50 a share") is the user's PRICE LIMIT, the most one share may cost: the Advisor already left out what costs more, and the Account Manager checks it again because the price
        moves: a purchase (approved or neutral) whose price is above it gets NO decision and the user is told (`decisions.over_limit_line`), every other decision carries it and the handshake refuses a fill above it. The neutral companies
        are all priced first when there is a limit, so the number asked for is filled from those under it. A limit is a price: never a size and never a rule about risk.

        `min_price` (entry 140; "the top 3 AI stocks over $20 a share") is the user's PRICE FLOOR, the same from the other side: a purchase priced under it gets no decision and the user is told, and the handshake refuses a fill under it.
        `shares_of` ({the stock as named: shares}) and `shares_each` (entry 140, C2; "buy 10 of MU if the recommendation passes", "5 shares of each of the top 3") are the shares the user TYPED with the recommendation: each approved (or neutral)
        purchase of such a stock is limited to them (the sizing's `wanted`, binding `asked`; the smaller of this and another limit wins), the cash is still the only other check, and a denied one buys nothing.

        RAISING FUNDS FOR THE SHARES THE USER TYPED (design choices entry 142, R10 and R11; `funding.py`). When EVERY approved (and neutral) purchase has typed shares (`shares_of`, `shares_each`) and what they cost, within the budget the user gave if any,
        is more than the AVAILABLE cash, the shortfall may be raised by selling holdings, and ONLY if the user permitted it: `raise_funds` (their words), the setting ALWAYS_RAISE_FUNDS (unless their words said not to sell), or a yes to ONE question
        in Ask mode (no answer is a no). Permitted, the purchases are sized as if the account held the money (each gets its typed shares: the budget is never exceeded), each decision carries `funding`, and the desk raises the whole shortfall in ONE
        RaiseFunds plan before the Trader sells and buys (`fund_decisions`). Not permitted (Auto mode without the words, a no), nothing is sold: the round robin buys fewer shares with the cash, as ever, and the user is told what to say. A purchase with no
        typed shares (sized to the cash) never triggers it, and a request that has one is not funded at all (the money raised would reach it).

        `top_pick` (entry 147): the user asked for THE most recommended company ("buy 5 shares of the most recommended shoe company", `intent.one_stock_asked`). When the Advisor recommended a company, that one company is the user's
        choice: it is bought if the Critic approves it and NOTHING is bought if the Critic denies it; no neutral company takes its place (entry 107's filling is for a number of companies, not for the one the user named by its rank),
        and the user is told why each neutral company was not bought. When the Advisor recommended none, the best neutral company is the most recommended one, as before.

        IGNORE CRITIC (design choices entry 148; the order says `ignore_critic`, the desk set it when it routed around the Critic): the Critic did not run, so every recommendation of the Advisor counts as approved (`_advisor_picks`; `origin: "advisor"`, no review): Auto
        trades them (`yes_auto`), Ask asks about each (a card that says the Critic did not review it) in the same proposal as the neutral purchases, and they are sized TOGETHER with those in ONE call, the Advisor's first. `count` keeps the number asked for (the Advisor's
        recommendations first, then the neutral companies). With the box off none of this is touched.

        Returns the SAME order, `decided` unless an answer is pending. Refuses (a message on the order) a profile that is not the portfolio's user's."""
        from prometheus.agents.account_manager import clarify as clarifying, decisions
        if each is not None and budget is not None:
            order.errors = [*order.errors, "an amount for each stock and one total (a budget) are two ways to say how much: give one of them"]
            return order
        approved = decisions.purchases_to_size(order, portfolio, now, max_price, min_price)      # the purchases the Critic's approvals make (a denied recommendation is not one; one priced above the user's limit is not either): [] unless the order is `reviewed`
        advisor_chosen, advisor_purchases = self._advisor_picks(order, approved, count, max_price, min_price)      # "Ignore Critic" (entry 148): the Advisor's recommendations count as approved, first by rank; [] with the box off
        names = neutral.neutral_candidates(order)
        known_prices: dict[str, Decimal] = {}
        if (max_price is not None or min_price is not None) and names:      # the user's price limit and floor (entries 138 and 140): price every neutral company first, so that the number asked for is filled from those inside them
            names, known_prices = self._neutral_under_limit(order, names, max_price, min_price)
        pick = top_pick and bool(order.recommendations)      # "the most recommended company" and the Advisor recommended one: that one or nothing (entry 147)
        chosen, unneeded = ([], names) if pick else neutral.choose(names, len(approved) + len(advisor_chosen), count)        # every neutral company counts as approved (entry 107), up to the number the request asked for, after the Advisor's own (entry 148)
        self._not_needed(order, unneeded, count, top=order.recommendations[0] if pick else None)
        neutral_prices = (known_prices or self._neutral_prices(order, chosen)) if chosen else {}
        neutral_purchases = neutral.purchases_for(order, chosen, neutral_prices, max((p.rank for p in [*approved, *advisor_purchases]), default=0))      # ranked after the approved ones: the sizing visits them in that order
        extra = [p for p in [*advisor_purchases, *neutral_purchases] if p.price is not None]      # asked about with the approved ones: ONE set of questions, over everything that is to be bought
        prices = self._held_prices(order, portfolio, now, bool(extra))
        wanted = None
        if each is not None:      # the user said the amount of EACH stock (entry 125): the plan of the answer "an amount for each company", with no question
            if approved or extra:
                plan = clarifying.plan_each_dollars(each, [*approved, *extra], portfolio.cash_avail)
                balance, budget, wanted = None, plan.budget, plan.wanted
                self.tell_user("", f"  How much: {plan.text}.")
        elif (clarify is not None and getattr(clarify, "available", lambda: True)() and (mode or self.profile.trade_mode) == "ask" and budget is None and not said_how_much
                and (decisions.has_purchase_to_size(order, now, max_price, min_price) or extra)):
            plan = self.clarify_sizing(order, portfolio, prices, clarify, now, balance, request_text, extra, max_price=max_price, min_price=min_price)
            if plan is not None:
                balance, budget, wanted = plan.balance, plan.budget, plan.wanted
        if shares_of or shares_each is not None:      # the shares the user typed with the recommendation (entry 140, C2): each such purchase is limited to them; the smaller of two limits on one purchase wins
            rows = [(p, decisions.company_of(order, p)) for p in [*approved, *(p for p in advisor_purchases if p.price is not None)]] + [(p, c.get("company")) for c, p in zip(chosen, neutral_purchases) if p.price is not None]
            typed = decisions.typed_wanted(rows, shares_of, shares_each)
            if typed:
                wanted = decisions.combine_wanted(wanted, typed)
                symbol_of = {p.record_id: p.symbol for p, _ in rows}
                self.tell_user("", "  How much, as you typed it: " + ", ".join(f"{sh.text(n)} {sh.noun(n)} of {symbol_of[record]}" for record, n in typed.items()) + " (if approved; the cash is the only other limit).")
        raising = None
        if (shares_of or shares_each is not None) and each is None and wanted:      # the shares the user typed may be paid for by selling holdings (entry 142, R10): only when the user permitted it
            raising = self._typed_funding(order, portfolio, [*approved, *extra], wanted, budget, prices, now, mode, approver, raise_funds, request_text)
        notes_before = len(order.data.notes) if order.data is not None else 0
        order = decisions.decide_trades(order, portfolio, self.profile, approver, now, mode, balance, budget, prices, wanted, neutral_purchases=list(zip(chosen, neutral_purchases)),
                                        fractional=self.fractional_split(balance), max_price=max_price, min_price=min_price, funding=raising, advisor_purchases=list(zip(advisor_chosen, advisor_purchases)))
        for note in (order.data.notes[notes_before:] if order.data is not None else []):      # what a price limit or floor left without a decision: told to the user (the notes of the neutral ones were told as they were made)
            for prefix in (decisions.OVER_LIMIT, decisions.UNDER_FLOOR):
                if note.startswith(prefix):
                    self.tell_user(f"      {note[len(prefix):]}")
        self._save_decisions(order)
        return order

    def _advisor_picks(self, order: WorkOrder, approved: list[sizing.Purchase], count: int | None, max_price: Decimal | None, min_price: Decimal | None) -> tuple[list[dict], list[sizing.Purchase]]:
        """(the Advisor's recommendations that are bought, their purchases) when the user chose to IGNORE THE CRITIC (entry 148; the order says `ignore_critic`), else ([], []): nothing changes with the box off. Every Advisor recommendation counts as approved and is
        priced the way a neutral company is: ONE quote order to the Data Clerk (`_neutral_prices`; with a price limit or floor `_neutral_under_limit`, which leaves out one priced outside it, tells the user and writes the note), so a recommendation with no
        price is decided `no_system_denied` (nothing is bought that cannot be priced). The number the request asked for (`count`) is kept: the best ranked come first, and a recommendation beyond it gets no decision and the user is told. Never raises."""
        records, _ = ignore_critic.recommendations_of(order)
        if not records:
            return [], []
        if max_price is not None or min_price is not None:
            records, prices = self._neutral_under_limit(order, records, max_price, min_price)
        else:
            prices = self._neutral_prices(order, records[:MAX_QUOTE_SYMBOLS])
        chosen, beyond = ignore_critic.choose(records, count)
        for record in beyond:
            line = ignore_critic.not_needed_line(record, count)
            self.tell_user(f"      {line}")
            if order.data is not None:
                order.data.notes = [*order.data.notes, ignore_critic.NOT_NEEDED + line]
        return chosen, ignore_critic.purchases_for(order, chosen, prices, max((p.rank for p in approved), default=0))

    def _typed_funding(self, order: WorkOrder, portfolio: Portfolio, rows: list[sizing.Purchase], wanted: dict[str, Decimal], budget: Decimal | None, prices: dict[str, Decimal], now, mode: str | None, approver,
                       raise_funds: bool, request_text: str) -> tuple[Decimal, str] | None:
        """(the money to raise, how the user permitted it) for the shares typed with a recommendation (R10 and R11), or None: nothing to raise (the cash covers it, or a purchase has no typed shares), the holdings could not pay for it, or the user
        did not permit selling (Auto mode without the words, a no, no answer); the user is told which. The shortfall is the cost of the typed shares of every purchase less the available cash, never more than the budget the user gave leaves."""
        rows = [p for p in rows if p.price is not None]
        left_out = [p for p in rows if p.record_id not in wanted]      # entry 143, P1: a purchase with NO typed shares is sized to what the cash leaves after the typed ones, never to money raised by selling
        rows = [p for p in rows if p.record_id in wanted]
        if not rows:
            return None
        need = sum((trade_amount(wanted[p.record_id], p.price) for p in rows), Decimal("0.00"))
        cap = need if budget is None else min(need, budget)
        shortfall = cap - portfolio.cash_avail
        if shortfall <= 0:
            return None
        mode = mode or self.profile.trade_mode
        feasible, note = sizing.funds_check(cap, portfolio, prices, eastern_date(now))
        if not feasible:
            self.tell_user(f"      Raising funds could not pay for the shares you typed: ${shortfall:,.2f} is short of the cash ${portfolio.cash_avail:,.2f}, and {note}. The cash buys what it can.")
            return None
        by = funding.permission(raise_funds, request_text)
        if by is None and mode == "ask" and approver is not None and getattr(approver, "available", lambda: True)():
            buys = [{"symbol": p.symbol, "shares": wanted[p.record_id], "price": p.price, "cost": trade_amount(wanted[p.record_id], p.price)} for p in rows]
            answer = self._ask_raise_funds(order, approver, funding.ask_card(portfolio.cash_avail, buys, shortfall, left_out=[p.symbol for p in left_out]))
            if answer is not True:
                self.tell_user("      Nothing is sold" + (": you did not want holdings sold to pay for it." if answer is False else ": there was no answer, and no answer is a no.") + " The cash buys what it can.")
                return None
            by = "asked"
        if by is None:
            self.tell_user(f"      The shares you typed cost ${cap:,.2f}, ${shortfall:,.2f} more than the cash ${portfolio.cash_avail:,.2f}, and nothing is sold unless you say so: {funding.NOT_SAID}. The cash buys what it can.")
            return None
        self.tell_user(f"      The shares you typed cost ${cap:,.2f}, ${shortfall:,.2f} more than the cash ${portfolio.cash_avail:,.2f}: {funding.how_words(by)}, so I will raise the money by selling some of your holdings first (RaiseFunds), and then buy.")
        if left_out:
            self.tell_user(f"      {funding.left_out_line([p.symbol for p in left_out])}")
        return shortfall, by

    def _neutral_under_limit(self, order: WorkOrder, names: list[dict], max_price: Decimal | None, min_price: Decimal | None = None) -> tuple[list[dict], dict[str, Decimal]]:
        """(the neutral companies priced inside the user's limit and floor, their prices) (entries 138 and 140, L4 and F2): ONE quote order for the companies, then each one above `max_price` or under `min_price` is left out with a line for
        the user and a note on the order's data (`decisions.OVER_LIMIT`, `decisions.UNDER_FLOOR`). A company with no price stays (it is `no_system_denied` when decided: nothing is bought that cannot be priced). Never raises (`_neutral_prices`)."""
        from prometheus.agents.account_manager import decisions
        from prometheus.common.work_order import MAX_QUOTE_SYMBOLS

        prices = self._neutral_prices(order, names[:MAX_QUOTE_SYMBOLS])
        kept = []
        for candidate in names:
            at = prices.get(candidate["symbol"])
            if at is not None and max_price is not None and at > max_price:
                note = decisions.OVER_LIMIT + decisions.over_limit_line(candidate["symbol"], at, max_price)
            elif at is not None and min_price is not None and at < min_price:
                note = decisions.UNDER_FLOOR + decisions.under_floor_line(candidate["symbol"], at, min_price)
            else:
                kept.append(candidate)
                continue
            self.tell_user(f"      {note.split(': ', 1)[1]}")
            if order.data is not None:
                order.data.notes = [*order.data.notes, note]
        return kept, {symbol: at for symbol, at in prices.items() if symbol in {c["symbol"] for c in kept}}

    def _neutral_prices(self, order: WorkOrder, names: list[dict]) -> dict[str, Decimal]:
        """{symbol: current price} of the neutral companies, from ONE quote order to the Data Clerk (`request_prices`). Never raises: a price that cannot be had is a message on the order and that company is left out (it is not bought)."""
        try:
            quoted = self.request_prices([c["symbol"] for c in names])
            quotes = dict(quoted.data.quotes) if quoted.data else {}
        except Exception as error:  # an outage, a missing key: the companies cannot be priced, and say so
            order.errors = [*order.errors, f"the current prices of {', '.join(c['symbol'] for c in names)} could not be had ({type(error).__name__})"]
            quotes = {}
        return _prices_of(quotes, [c["symbol"] for c in names])

    def _not_needed(self, order: WorkOrder, unneeded: list[dict], count: int | None, top: dict | None = None) -> None:
        """Tell the user which neutral companies the number they asked for did not need (entry 107): one line each (`neutral.not_needed_line`), on the screen and as a note on the order's data. Such a company gets NO decision.
        `top`: the Advisor's recommendation when the user asked for THE most recommended company (entry 147): the line says that it is the one, whatever the Critic said of it."""
        for candidate in unneeded:
            line = neutral.not_top_pick_line(candidate, top) if top is not None else neutral.not_needed_line(candidate, count)
            self.tell_user(f"      {line}")
            if order.data is not None:
                order.data.notes = [*order.data.notes, neutral.NOT_NEEDED + line]

    @traced("account_manager.request_symbols")
    def request_symbols(self, stocks: list[str]):
        """Ask the Data Clerk for the company name and the symbol of each stock (a ticker in capitals, or a company's name as the user typed it): a `lookup_symbol` work order, answered by `DataClerk.request_symbols`
        (the symbol directory, with exact matching, never a guess; for a NAME the directory does not have, the Data Clerk's model names its listed owner and the directory gives the symbol: entry 101). Returns the order,
        `data_ready` with `data.companies`, or `data_failed` with the Data Clerk's own message for a name it did not find (its row has no symbol), one that fits several listings or a lookup that failed (no row). Beside `request_prices`."""
        order = new_work_order(f"Company names and symbols of {', '.join(stocks)}", requester=REQUESTER, action_type="lookup_symbol", subject=list(stocks))
        return self.clerk.request_symbols(order)

    @traced("account_manager.name_orders")
    def name_orders(self, order: WorkOrder, orders, typed: str, now=None, approver=None, mode: str | None = None) -> "direct.Naming":
        """A DIRECT ORDER skips the Data Clerk's node, so the Account Manager gets the company name and the symbol of each stock the user named itself (design choices entries 88 and 101): it asks the Data Clerk
        (`request_symbols`) and APPENDS the answer to the work order (`direct.name_the_stocks`: `data.companies`, the symbols as the subject, and the user's own orders in `direct_orders`). `orders` are the form's
        (`intent.orders`), `typed` the user's words. Returns a `direct.Naming`: the problems (empty when the work order now holds every stock that stays), the orders that stay and the ones that were left out. Never raises.

        WHAT HAPPENS TO A STOCK THE DATA CLERK DID NOT FIND (entry 102; Jacob: *"Auto mode will not ask any prompts ... Remove missing (not found / unmatched) companies if in auto mode"* and, in Ask mode, *"ask the user if they
        want to ignore the missing companies and continue"*). Every stock is found, not listed (the directory answered and has no such company) or not answered (the lookup failed: never "not found").
          - a BLOCKING issue stops the run in either mode: a stock whose lookup failed, nothing found at all, or two orders for the same stock. `problems` are the Data Clerk's reasons, and the run's last message asks the user to
            clarify the request (`direct.CLARIFY`).
          - some found and some not listed, AUTO mode (`mode`, else the profile's trade mode): the missing stocks are dropped WITHOUT a question, and the user is told which and why (`tell_user`, and `order.data.notes`).
            The approver is not called and the model is not asked.
          - the same in ASK mode: the user is asked about each missing stock (`missing.ask_about_each`: the model words each question, code checks it, `approver` answers yes or no). A yes drops that stock; a no, no answer or
            no approver stops the order, and nothing is dropped on the user's behalf."""
        from prometheus.agents.account_manager import direct, missing
        try:
            found = self.request_symbols([o.named for o in orders])
        except Exception as error:  # a missing key, an outage: nothing is guessed, the order is not made
            return direct.Naming(problems=[f"the Data Clerk could not be asked for the symbols ({type(error).__name__})"])
        sort = direct.sort_names(orders, found)
        dropped: list[dict] = []
        if sort.listed and sort.missing and not sort.unanswered and found.data is not None:
            mode = mode or self.profile.trade_mode
            kept = missing.kept_of(sort.listed)
            reasons = [(order_.named, missing.reason_of(row)) for order_, row in sort.missing]
            if mode == "auto":
                dropped = [{"typed": " ".join(named.split()), "reason": reason, "mode": "auto"} for named, reason in reasons]
            else:
                asked = missing.ask_about_each(self.model, sort.missing, kept, typed, approver, load_account_manager_config())
                self.last_missing = {"cards": asked.cards, "wordings": [{"source": w.source, "why": w.why, "calls": w.calls, "tokens_in": w.tokens_in, "tokens_out": w.tokens_out} for w in asked.wordings], "stopped": asked.stopped}
                if asked.stopped:
                    return direct.Naming(problems=[*found.errors, asked.stopped])
                dropped = asked.dropped
        problems = direct.name_the_stocks(order, orders, typed, now or trading_now(), found, dropped)
        if problems:
            return direct.Naming(problems=problems)
        stays = [o for o, _ in sort.listed]
        goes_on = missing.and_list([e["symbol"] for e in order.direct_orders])
        for d in dropped:
            self.tell_user(f"      {d['typed']} was not found ({d['reason']}) and was left out" + (" (Auto mode asks nothing)" if d["mode"] == "auto" else " at your word") + f"; the order goes on with {goes_on}")
        return direct.Naming(orders=stays, dropped=dropped)

    @traced("account_manager.name_the_stocks")
    def name_the_stocks(self, order: WorkOrder, orders, typed: str, now=None, approver=None, mode: str | None = None) -> list[str]:
        """`name_orders`, answering only with its problems (empty when the work order now holds every stock that stays): the older call, for a caller that does not need to know which stocks were left out."""
        return self.name_orders(order, orders, typed, now, approver, mode).problems

    @traced("account_manager.decide_direct")
    def decide_direct(self, order: WorkOrder, portfolio: Portfolio, now=None, mode: str | None = None, approver=None, clarify=None, said_how_much: bool = False, request_text: str = "",
                      balance: str | None = None, budget: Decimal | None = None, raise_funds: bool = False) -> WorkOrder:
        """The trade decisions for a DIRECT ORDER (design choices entries 87 and 88): the user named the trades, so no recommendation was made and the Critic was not asked (`direct.py`). The current price of each symbol is
        asked of the Data Clerk (one quote order: `request_prices`); each of `order.direct_orders` is then decided at its price: `yes_user` (the user's own order is the approval; the user is not asked again), or `no_funds`
        (the cash does not pay for the shares in full, or the account does not hold them: it is never made smaller), or `no_system_denied` (no price). When the user NAMED A COMPANY, the symbol is the Data Clerk's: in Ask mode
        `approver` (the callable that asks the user: True, False, None for no answer) confirms the symbol, the company, the shares and the amount first (no answer: `pending_user`, not traded); in Auto mode the symbol is used
        without asking. NO LIMIT OF THE PROFILE EXISTS (entry 91): the profile is read for its user and its trade mode, nothing else. A connected account book saves the decisions. `mode` overrides the profile's. Returns the SAME
        order, `decided`. A price that cannot be had is a message on the order; a profile that is not the portfolio's user's is refused (a message and nothing decided).

        A PURCHASE THE USER NAMED WITH NO NUMBER ("I want to buy MSFT", design choices entry 98) is still a direct order: no recommendation. In ASK mode, with `clarify` (the channel to the user) and no dollars or "as much as I can" in
        the request (`said_how_much`), the Account Manager first ASKS for the number with the same bounded questions as for any purchase (`clarify_sizing`); then it shows the order (the shares and the amount) and the user says yes
        or no. In AUTO mode nothing is asked: the funds are spent as the user's words say (`balance`, `budget`; round-robin by default: one stock gets what the cash allows). A SALE with no number is not placed: say how many.

        THE DOLLARS THE USER WROTE FOR A STOCK ARE ITS LIMIT (design choices entry 122: "Buy $3,000 of XOM and $3,000 of MSFT" spends $3,000 on each and $6,000 in all, never the whole cash): see `_size_direct`. A purchase with no number
        in a request where another purchase has one is asked about in Ask mode and is not placed in Auto mode (the cash is never its default).

        A PURCHASE THE CASH DOES NOT COVER (design choices entry 142; `funding.py`, design/direct_raise_funds_spec.md R1 to R8, R12). The direct buys of the request that are placed now (typed shares, the dollars of a stock, a total) cost
        more than the AVAILABLE cash: the shortfall. Holdings are sold to pay for it ONLY when the user permitted it: `raise_funds` (the form's: their words said "raise funds if needed"), the setting ALWAYS_RAISE_FUNDS (unless their words
        said not to sell), or, in Ask mode, a yes to ONE question (the card `raise_funds`; no answer is a no). Auto mode asks nothing: without the words it does not sell and says what to add. Permitted, each such purchase is decided
        `yes_user` as the user typed it (never cut) with `funding` on the decision, and the desk runs RaiseFunds before the Trader (`fund_decisions`); a purchase in dollars is sized with the money to be raised, its dollars the most it may cost.
        Not permitted, a purchase of typed shares is `no_funds` with the reason, and one in dollars is sized to the cash as before. If even selling every holding that may be sold could not pay for it (R8) it is `no_funds` too, naming
        the shortfall and what the holdings could raise."""
        from prometheus.agents.account_manager import direct
        now = now or trading_now()
        prices = self._direct_prices(order)
        sales, after = self._own_sales(order, portfolio, prices, now)      # entry 143, S1: the user's own sale in this request pays for the purchases too (it is placed first)
        proceeds = sum((s["amount"] for s in sales), Decimal("0.00"))
        by, shortfall, why_not = self._direct_funding(order, after, prices, now, mode, approver, raise_funds, request_text, budget, sales)
        sized = self._size_direct(order, after, prices, now, mode, clarify, said_how_much, request_text, balance, budget, extra_cash=shortfall if by else Decimal("0.00"))
        waiting_by, waiting_why = self._waiting_funding(order, portfolio, prices, now, raise_funds, request_text)      # entry 143, W-F: a waiting purchase the free cash does not cover, paid for by selling when its price is reached
        order = direct.decide_direct(order, portfolio, self.profile, mode, prices, now, load_account_manager_config().direct.valid_minutes, approver, sized, fractions=self._book_holds_fractions(),
                                     funding=self._funding_map(order, prices, budget, shortfall, by) if by else None, why_not={**why_not, **waiting_why}, sale_proceeds=proceeds, waiting_funding=waiting_by)
        self._place_waiting(order, now)      # an order whose limit the price has not reached WAITS as a day order, its cash or shares set aside (entry 140)
        self._save_decisions(order)
        return order

    def _place_waiting(self, order: WorkOrder, now: datetime) -> None:
        """Place in the account book every direct order that waits (design choices entry 140, W1 to W4): each decision `direct.decide_direct` gave a `waiting` plan (the order's limit is not reached, the user confirmed it where Ask mode asks) becomes an OPEN
        ORDER, with what it needs SET ASIDE, in ONE transaction that locks the account (`AccountStore.place_open_order`: the book refuses a placement the available cash or shares do not cover, whatever was read before). The decision then carries the
        order's number (`open_order_id`) and says it WAITS, and the user is told (`tell_user`): "Waiting: buy 10 MU at $90.00 or less (now $95.12), $900.00 set aside, until 4:00 pm ET today." Without a saved account (a portfolio file) nothing can wait, and the
        decision says so. A placement the book refused is `no_funds` with its sentence. Never raises for a database failure: the order is then not placed, and the decision says so."""
        for entry, decision in zip(order.direct_orders, order.trade_decisions):
            plan = decision.get("waiting")
            if not isinstance(plan, dict) or decision.get("open_order_id") is not None:
                continue
            decision.pop("waiting", None)
            if self.store is None or self.account_id is None:
                decision.update(will_trade="no_system_denied", funds_avail=False, shares=Decimal(0), est_cost=Decimal("0.00"),
                                reason=f"{entry['symbol']} is at ${plan['price']:,.2f} a share, {'above' if plan['side'] == 'buy' else 'under'} your limit of ${plan['limit']:,.2f}: not placed. An order that waits for its price needs the saved account book to set its cash or shares aside, and this run has none")
                self.tell_user(f"      {decision['reason']}")
                continue
            try:
                placed = self.store.place_open_order(self.account_id, request_id=order.request_id, record_id=entry["record_id"], symbol=entry["symbol"], company=entry.get("company"), side=plan["side"],
                                                     limit_price=plan["limit"], request=entry["request"], placed_at=now, expires_at=_moment(plan["expires_utc"]),
                                                     shares=plan["shares"], dollars=plan["dollars"], raise_funds_by=plan.get("funded_by"))
            except ValueError as error:      # the book's own refusal: what it would set aside is not available (another order took it, or the cash moved)
                decision.update(will_trade="no_funds", funds_avail=False, shares=Decimal(0), est_cost=Decimal("0.00"), reason=" ".join(str(error).split())[:300] + ": the order cannot wait, and nothing is placed")
                self.tell_user(f"      {decision['reason']}")
                continue
            except Exception as error:      # a database or network failure: nothing is known to be set aside, so nothing is placed
                decision.update(will_trade="no_system_denied", funds_avail=False, shares=Decimal(0), est_cost=Decimal("0.00"), reason=f"the account book could not be asked to set the order aside ({type(error).__name__}): nothing is placed")
                self.tell_user(f"      {decision['reason']}")
                continue
            decision["open_order_id"] = placed.order_id
            decision["waiting_utc"] = plan["expires_utc"]
            orders.audit({"event": "placed", "text": decision["reason"], "order": placed.as_dict()})
            self.tell_user(f"      {decision['reason']} (order #{placed.order_id}: it can be cancelled on the page, or with `python -m prometheus.agents.account_manager.orders --account {self.account_id} --cancel {placed.order_id}`)")

    @traced("account_manager.fill_open_order")
    def fill_open_order(self, open_order, price: Decimal, trader, now: datetime) -> WorkOrder:
        """FILL a waiting order whose limit the quote has reached (design choices entry 140, W5): the SAME funnel as any direct order, no second booking path. The order becomes a one-order work order (the user's own record, with the words they
        typed and the limit they wrote), `direct.decide_direct` decides it at `price` (`yes_user`: the user's own order is the approval, and the limit is checked again by the handshake at the fill price), and `perform_trades` hands it to `trader` (the
        paper venue, no model): `approve_trade` reads the account with THIS order's own hold available to it and to nothing else (`_portfolio`), the Trader fills at the quote, `verify_fill` checks the fill, and `record_fill` books it in the book's one
        transaction (`book_trade(order_id=...)`: the order becomes `filled` and its hold is gone in the same commit; a second fill is refused). A purchase in dollars buys what its dollars pay for at the fill price (`shares.buys`; whole shares when the book
        cannot hold a fraction), never more than the order set aside. Returns the work order. A fill that fails is a `rejected` trade (or a decision that is not `yes_user`) on it: the caller closes the order (`AccountStore.reject_open_order`), nothing is retried."""
        from prometheus.agents.account_manager import direct
        self._filling = open_order
        try:
            work = new_work_order(open_order.request, requester=REQUESTER, action_type="trade")
            fractions = self._book_holds_fractions()
            in_dollars = open_order.dollars is not None
            shares = open_order.shares if not in_dollars else (sh.buys(open_order.dollars, price) if fractions else Decimal(int(open_order.dollars // price)))      # an order in dollars: what its dollars buy at THIS price
            entry = {"record_id": open_order.record_id, "side": open_order.side, "symbol": open_order.symbol, "company": open_order.company or open_order.symbol, "shares": shares, "request": open_order.request,
                     "requested_utc": open_order.placed_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "typed_company": None, "how_many": "dollars" if in_dollars else "typed", "found_by": None,
                     **({"dollars": open_order.dollars} if in_dollars else {}), ("max_price" if open_order.side == "buy" else "min_price"): open_order.limit_price}
            work.direct_orders, work.subject, work.data = [entry], [open_order.symbol], DataResult()
            sized = {open_order.record_id: {"shares": shares, "how": "dollars", "pot": open_order.dollars, "dollars": open_order.dollars if fractions else None, "allocation": "round_robin", "binding": None}} if in_dollars and shares > 0 else None
            portfolio = self._portfolio()      # with this order's own hold available to it
            fund_map, why_not = self._funding_at_fill(work, open_order, portfolio, shares, price, now)      # entry 143, W-F4: the price is reached and the cash does not cover it: the user's permission, given when the order was placed, is used NOW
            work = direct.decide_direct(work, portfolio, self.profile, "auto", {open_order.symbol: price}, now, load_account_manager_config().direct.valid_minutes, None, sized, fractions=fractions, funding=fund_map or None, why_not=why_not)
            self._save_decisions(work)
            funds = self.fund_decisions(work, portfolio, now) if fund_map else None      # RaiseFunds, once, for the shortfall at this price; when it cannot pay for it the decision becomes `no_funds` and nothing is sold
            if not any(d.get("will_trade") in ("yes_auto", "yes_user") for d in work.trade_decisions):
                return work
            return self.perform_trades(work, portfolio, trader, now, funding=funds, book=True)
        finally:
            self._filling = None

    def _funding_at_fill(self, work: WorkOrder, open_order, portfolio: Portfolio, shares: Decimal, price: Decimal, now) -> tuple[dict[str, dict], dict[str, str]]:
        """({record_id: `funding.decision_funding`}, {record_id: why not}) for the fill of a waiting PURCHASE that the user permitted to raise funds (design choices entry 143, W-F4; `open_order.raise_funds_by`, "words" or "setting", written when it was
        placed). Its price is reached: if the cash it may use now (the free cash, and its own hold, which `portfolio` already counts) pays for the shares at this price, nothing is raised and it fills as any order; if not, the shortfall is what
        they cost less that cash, the arithmetic says whether the holdings that may be sold could pay for it (R8: when they could not, nothing is sold and the order is closed with the reason), and the decision carries `funding` so that
        `fund_decisions` runs RaiseFunds and the Trader sells first and buys after, one trade at a time. Nobody is asked (nobody is there): the permission is the user's own, given when they placed the order. Never raises."""
        if open_order.side != "buy" or open_order.raise_funds_by is None:
            return {}, {}
        cost = trade_amount(shares, price)
        short = cost - portfolio.cash_avail
        if short <= 0:
            return {}, {}
        feasible, note = self._holdings_could_raise(work, portfolio, cost, now)
        if not feasible:
            line = f"raising funds could not pay for it: ${short:,.2f} is short of the cash ${portfolio.cash_avail:,.2f} at ${price:,.2f} a share, and {note}"
            self.tell_user(f"      Order #{open_order.order_id}: {line}")
            return {}, {open_order.record_id: line}
        self.tell_user(f"      Order #{open_order.order_id}: the price is reached and the cash ${portfolio.cash_avail:,.2f} is ${short:,.2f} short of {sh.text(shares)} {open_order.symbol} at ${price:,.2f}: {funding.how_words(open_order.raise_funds_by)}, so I raise the money by selling some of your holdings first (RaiseFunds), and then buy.")
        return {open_order.record_id: funding.decision_funding(short, open_order.raise_funds_by)}, {}

    def _size_direct(self, order: WorkOrder, portfolio: Portfolio, prices: dict[str, Decimal], now, mode: str | None, clarify, said_how_much: bool, request_text: str, balance: str | None, budget: Decimal | None,
                     extra_cash: Decimal = Decimal("0.00")) -> dict[str, dict]:
        """The number of shares of each PURCHASE the user named with no number of shares: {record_id: {"shares", "how", "pot", "dollars", ...}} ("dollars": the money the user wrote for THAT stock, entry 122; "asked": the user answered
        the Account Manager's questions; "budget": the one total the user gave; "cash": all the cash, spread round-robin over the stocks named: entry 106; "unsaid": see below, with no shares). Nothing else bounds it (no limit of any
        profile). Never raises: whatever fails leaves the order unsized, and `direct.decide_direct` refuses it with the reason.

        THE USER'S OWN DOLLARS FOR A STOCK ARE ITS LIMIT (entry 122; "Buy $3,000 of XOM and $3,000 of MSFT"). A purchase with `dollars` is sized as an amount for EACH stock, exactly as the answer "$3,000 in each" of entry 111, M3: `wanted` is
        what its dollars buy (a fraction when the account book holds one, else the whole shares), and it is an order in dollars (`dollars` on the decision: the handshake re-sizes it at the fill price and never past it). The POT of the
        run is the sum of those dollars (never above the cash). The cash is the only other limit.

        A PURCHASE WITH NO NUMBER IN A REQUEST WHERE ANOTHER PURCHASE HAS ONE ("Buy $2,000 of AAPL and MSFT", with no total and no "as much as I can"; entry 122, D6) is never given the cash by default: in Ask mode the Account Manager asks
        how much of it (`clarify_sizing`, about that purchase alone) and the answer is its size; with no answer, and in Auto mode, it is `unsaid` and `direct.decide_direct` does not place it ("say how much of MSFT"), the others go ahead.
        When both kinds are sized, every decision carries the SAME pot (a run has one): the sum of the pots of the two.

        `extra_cash` (entry 142, R1): the money the Account Manager is about to RAISE by selling holdings, which the user permitted: the purchases are sized as if the account held it (the pot is the dollars the user wrote, never more),
        and the handshake still reads the real cash at every buy, so nothing is ever spent that is not there."""
        from prometheus.agents.account_manager import direct
        if extra_cash:
            portfolio = portfolio.model_copy(update={"cash_avail": portfolio.cash_avail + extra_cash})      # sized as if the money were there; the work order's own portfolio is not changed

        entries = [e for e in order.direct_orders if isinstance(e, dict) and e.get("shares") is None and e.get("side") == "buy" and prices.get(e.get("symbol")) and not direct.not_now(e, prices.get(e.get("symbol")))]      # one priced above its ceiling or under its floor is not placed now (entries 138 and 140: it waits, or is not placed), so it takes no part of the money
        if not entries:
            return {}
        mode = mode or self.profile.trade_mode
        purchases = {e["record_id"]: sizing.Purchase(record_id=e["record_id"], symbol=e["symbol"], action="open", price=prices[e["symbol"]], rank=n) for n, e in enumerate(entries, 1)}
        fractions = self.fractions_on("the money is spent in whole shares")      # entry 111, M3: whole shares, and a fraction of a share with what is left, when the account book can hold one
        own, bare = [e for e in entries if e.get("dollars") is not None], [e for e in entries if e.get("dollars") is None]
        anchored = bool(own) or any(e.get("side") == "buy" and e.get("shares") is not None for e in order.direct_orders if isinstance(e, dict))      # another purchase of the request carries the user's number
        out: dict[str, dict] = {}
        pots: list[Decimal] = []
        if own:
            given = self._size_dollars_each(order, portfolio, [purchases[e["record_id"]] for e in own], {e["record_id"]: e["dollars"] for e in own}, fractions)
            out.update(given)
            pots += [v["pot"] for v in given.values()][:1]
        if bare:
            given = self._size_unnumbered(order, portfolio, [purchases[e["record_id"]] for e in bare], now, mode, clarify, said_how_much, request_text, balance, budget, fractions, anchored)
            out.update(given)
            pots += [v["pot"] for v in given.values() if v.get("pot") is not None][:1]
        if len(pots) > 1:      # two groups, one pot: a run has one (`work_order.pot_reasons`), and it is never above the cash
            run_pot = sizing.pot_of(portfolio.cash_avail, sum(pots, Decimal("0.00")))
            out = {record: {**v, "pot": run_pot} if v.get("pot") is not None else v for record, v in out.items()}
        return out

    def _size_dollars_each(self, order: WorkOrder, portfolio: Portfolio, purchases: list[sizing.Purchase], dollars: dict[str, Decimal], fractions: bool) -> dict[str, dict]:
        """The purchases the user gave dollars for, each as an amount for EACH stock (entry 122, rule D3): `wanted` is `shares.buys(its dollars, its price)` with fractions, else the whole shares its dollars pay for; the budget of the
        sizing is the sum of the dollars, so the pot is that sum (never above the cash). Every purchase that gets shares carries its dollars (with fractions: an order in dollars, `how: "dollars"`). Never raises (see `_size_direct`)."""
        try:
            amount = {p.record_id: money_door(dollars[p.record_id]) for p in purchases}
            wanted = {p.record_id: (sh.buys(amount[p.record_id], p.price) if fractions else Decimal(int(amount[p.record_id] // p.price))) for p in purchases}
            sized = sizing.size_purchases(purchases, portfolio.cash_avail, allocation="round_robin", budget=sum(amount.values(), Decimal("0.00")), wanted=wanted, fractional=fractions, fractional_leftover=fractions)
        except ValueError as error:      # only a program gets these wrong: a message on the order, nothing is bought
            order.errors = [*order.errors, f"the amounts you wrote could not be sized: {error}"]
            return {}
        return {p.record_id: {"shares": s.shares, "how": "dollars", "pot": s.pot, "dollars": s.dollars if fractions else None, "allocation": s.allocation, "binding": s.binding} for p, s in zip(purchases, sized) if s.shares > 0}

    def _size_unnumbered(self, order: WorkOrder, portfolio: Portfolio, purchases: list[sizing.Purchase], now, mode: str, clarify, said_how_much: bool, request_text: str, balance: str | None, budget: Decimal | None,
                         fractions: bool, anchored: bool) -> dict[str, dict]:
        """The purchases the user named with no number and no dollars (entry 98): the one total they gave, all the cash (round robin), or what they answer when asked (Ask mode). When another purchase of the request carries a number
        and there is no total and no "as much as I can" (`anchored`; entry 122, D6), the cash is not spent on it by default: it is only sized by an answer, and otherwise it is `unsaid`: {"shares": None, "how": "unsaid"}."""
        wanted, how, in_dollars, answered = None, "budget" if budget is not None else "cash", True, False
        if mode == "ask" and clarify is not None and getattr(clarify, "available", lambda: True)() and budget is None and not said_how_much:
            plan = self.clarify_sizing(order, portfolio, self._held_prices(order, portfolio, now, True), clarify, now, balance, request_text, purchases, fractional=fractions)
            if plan is not None:
                balance, budget, wanted, how, in_dollars, answered = plan.balance, plan.budget, plan.wanted, "asked", wanted_in_dollars(plan), True
        if anchored and budget is None and not answered and not said_how_much:
            return {p.record_id: {"shares": None, "how": "unsaid", "pot": None, "dollars": None, "allocation": None, "binding": None} for p in purchases}
        try:
            sized = sizing.size_purchases(purchases, portfolio.cash_avail, allocation=balance or "round_robin", budget=budget, wanted=wanted, fractional=fractions, fractional_leftover=fractions)
        except ValueError as error:  # only a program gets these wrong: a message on the order, nothing is bought
            order.errors = [*order.errors, f"the purchases you named could not be sized: {error}"]
            return {}
        # what the decision needs of the sizing: the shares (the ESTIMATE of an order in dollars), where the number came from, the POT of the run (M1), and the dollars of an ORDER IN DOLLARS (M2: money from the user's words: a total, the
        # cash, or an amount for each stock; a number of shares the user answered stays an order in shares)
        return {p.record_id: {"shares": s.shares, "how": how, "pot": s.pot, "dollars": s.dollars if in_dollars else None, "allocation": s.allocation, "binding": s.binding}
                for p, s in zip(purchases, sized) if s.shares > 0}

    def _direct_prices(self, order: WorkOrder) -> dict[str, Decimal]:
        """{symbol: current price} of the tickers the user named, from ONE quote order to the Data Clerk. Never raises: a price that cannot be had is left out (that order is then not decided as a trade) and the order says so."""
        symbols = list(dict.fromkeys(entry["symbol"] for entry in order.direct_orders if isinstance(entry, dict) and isinstance(entry.get("symbol"), str)))
        if not symbols:
            return {}
        try:
            quoted = self.request_prices(symbols)
            quotes = dict(quoted.data.quotes) if quoted.data else {}
        except Exception as error:  # an outage, a missing key: the orders cannot be priced, and say so
            order.errors = [*order.errors, f"the current prices of {', '.join(symbols)} could not be had ({type(error).__name__})"]
            return {}
        prices = _prices_of(quotes, symbols)
        missing = [symbol for symbol in symbols if symbol not in prices]
        if missing:
            order.errors = [*order.errors, f"no current price for {', '.join(missing)}"]
        return prices

    @traced("account_manager.clarify_sizing")
    def clarify_sizing(self, order: WorkOrder, portfolio: Portfolio, prices: dict[str, Decimal], channel, now=None, balance: str | None = None, request_text: str = "", extra: list[sizing.Purchase] | None = None,
                       fractional: bool = False, max_price: Decimal | None = None, min_price: Decimal | None = None):
        """Ask the user how much to buy (design choices entry 85, `clarify.py`): a bounded ReAct loop with this agent's model. It phrases at most three questions through `channel`, reads the user's answer (a percentage
        of the account, dollars, shares, or "as much as I can") into a form, and CODE checks the form against what the user typed and works out the whole shares (the only thing that can cut it is the cash: there is no
        limit of the profile to raise, entry 91). Returns a `clarify.SizingPlan` (what `decisions.decide_trades` sizes under), or None when nothing could be worked out (no purchase, a model that cannot be reached, nonsense,
        or nobody answering): the order then says why and the default size (all the cash, a share of each company in turn) is shown. Never raises. `last_sizing` says what it cost and what was asked and answered."""
        from prometheus.agents.account_manager import clarify as clarifying, decisions
        config = load_account_manager_config()
        self.last_sizing = {"model": getattr(self.model, "model_name", None), "calls": 0, "questions": [], "answers": [], "problems": [], "choice": None, "plan": None, "why": ""}
        try:
            purchases = [*decisions.purchases_to_size(order, portfolio, now, max_price, min_price), *(extra or [])]      # the Critic's approved purchases (none outside the user's price limit and floor), and the neutral ones that count as approved (entry 91): one set of questions
            if not purchases:
                return None
            equity, _ = sizing.equity_of(portfolio, prices)
            cash = portfolio.cash_avail
            facts = clarifying.facts_for(purchases, [sizing.size_purchases([p], cash, config)[0] for p in purchases], cash, equity, request_text, balance, config)      # what the cash buys of EACH company on its own (one at a time: not how the money will be spread)
            outcome = clarifying.run_loop(self.model, facts, purchases, cash, equity, channel, config, balance)
            self.last_sizing.update(calls=outcome.calls, questions=outcome.questions, answers=outcome.answers, problems=outcome.problems, tokens_in=outcome.tokens_in, tokens_out=outcome.tokens_out, why=outcome.why)
            if outcome.choice is None:
                raise ValueError(outcome.why or "no amount was given")
            plan = clarifying.plan_choice(outcome.choice, purchases, equity, cash, prior_balance=balance, fractional=fractional)
        except AssertionError:
            raise
        except Exception as error:   # whatever went wrong, the default size is shown and the user is told: never a stop
            why = " ".join(str(error).split())[:200] if isinstance(error, ValueError) else f"{type(error).__name__}"
            self.last_sizing["why"] = self.last_sizing["why"] or why
            order.errors = [*order.errors, f"how much to buy could not be worked out with you ({why}): the default size is shown (all your cash, a share of each company in turn), and you can still say no to each trade"]
            self.tell_user("", f"  I could not work out how much you want ({why}). The default size is shown below (all your cash, a share of each company in turn); you can still say no to each trade.")
            return None
        self.last_sizing.update(choice=outcome.choice.model_dump(), plan=plan.model_dump())
        self.tell_user("", f"  How much: {plan.text}.", *[f"      {note}" for note in plan.notes])
        return plan

    def _held_prices(self, order: WorkOrder, portfolio: Portfolio, now=None, neutral_waiting: bool = False) -> dict[str, Decimal]:
        """{symbol: current price} of the symbols `portfolio` holds, from ONE quote order to the Data Clerk, and only when something needs it: the order is waiting to be decided, an approved purchase that has not run out
        is on it, and the portfolio holds something (cash alone needs no quote). Never raises: a price that cannot be had is left out (the equity counts that holding at its cost basis) and the order says so."""
        from prometheus.agents.account_manager import decisions
        if not portfolio.positions or not (neutral_waiting or decisions.has_purchase_to_size(order, now)):
            return {}
        try:
            quoted = self.request_prices(portfolio.symbols)
            quotes = dict(quoted.data.quotes) if quoted.data else {}
        except Exception as error:  # an outage, a missing key: the size is then worked out at cost basis, and says so (never a stop)
            order.errors = [*order.errors, f"the current prices of the holdings could not be had ({type(error).__name__}): they count at cost basis in the equity"]
            return {}
        prices = _prices_of(quotes, portfolio.symbols)
        missing = [symbol for symbol in portfolio.symbols if symbol not in prices]
        if missing:
            order.errors = [*order.errors, f"no current price for {', '.join(missing)}: counted at cost basis in the equity"]
        return prices

    def record_user_decision(self, order: WorkOrder, record_id: str, approve: bool, now=None) -> WorkOrder:
        """The user's later answer to a decision that was `pending_user`: yes, no, or `no_expired` when the approval has run out by now. A connected account book is updated with the same answer."""
        from prometheus.agents.account_manager import decisions
        order = decisions.record_user_decision(order, record_id, approve, now)
        self._save_decisions(order)
        return order

    # ---- the account book (Postgres). None of this calls a model or a quote.

    @staticmethod
    def _check_book(store, account_id) -> None:
        """The rule of the account book's two arguments: both or neither (`__init__`), and an account number that is a positive whole number."""
        if account_id is not None and (isinstance(account_id, bool) or not isinstance(account_id, int) or account_id < 1):
            raise ValueError("account_id must be a positive whole number")
        if (store is None) != (account_id is None):
            raise ValueError("pass store and account_id together (the Postgres account book), or neither")

    def connect(self, store, account_id: int) -> None:
        """Attach the Postgres account book after construction (the desk builds the manager first and learns the account afterwards): the same checks as `__init__`, and both arguments are required. A profile this manager
        read from a file nobody gave it (the default JSON file) is forgotten, so the profile becomes the account's own row, read when first needed; a profile that was given (an object, or a path not read yet) stays.
        Nothing is read from the database here."""
        if store is None or account_id is None:
            raise ValueError("connect needs both the store and the account_id (the Postgres account book)")
        self._check_book(store, account_id)
        self.store, self.account_id = store, account_id
        if self._profile_from_file:
            self._profile, self._profile_from_file = None, False

    def _book(self):
        if self.store is None or self.account_id is None:
            raise ValueError("this Account Manager has no saved account (pass store= and account_id=; see sql/prometheus_accounts.sql)")
        return self.store

    def _forget_saved_profile(self) -> None:
        if self.account_id is not None:
            self._profile = None

    def _save_decisions(self, order: WorkOrder) -> None:
        if self.store is None or self.account_id is None or not order.trade_decisions:
            return
        holder = self.store.get_user(self.account_id)
        if self.profile.user != holder.display_name:
            order.errors = [*order.errors, f"the decisions were not saved: the profile is for {self.profile.user!r} and the account is {holder.display_name!r}"]
            return
        self.store.save_decisions(self.account_id, order.trade_decisions)

    @property
    def users(self) -> UserDirectory:
        """The holder's name, account number, and the fields a later screen will show."""
        self._book()
        return UserDirectory(self.store, self.account_id, on_change=self._forget_saved_profile)

    def add_funds(self, amount: Decimal | str | int, when: date | None = None) -> Decimal:
        """Add settled cash. `amount` is a decimal number of dollars (a float is refused). Returns the cash balance after the deposit."""
        return self._book().add_funds(self.account_id, amount, when)

    def remove_funds(self, amount: Decimal | str | int, when: date | None = None) -> Decimal:
        """Take settled cash out of the account (a withdrawal): `amount` is a decimal number of dollars (a float is refused) and the account must hold it. Returns the cash balance after it."""
        return self._book().remove_funds(self.account_id, amount, when)

    def list_portfolio(self) -> Portfolio:
        """The saved holdings and the settled cash, as the same Portfolio the rest of the desk already reads."""
        return self._book().list_portfolio(self.account_id)

    def history(self) -> dict:
        """The account's dated trades and cash ledger as the book keeps them (read only): `{trades: [...], cash: [...]}`, oldest first. The Proving Ground's live monitor reads it; nothing is written."""
        return self._book().history(self.account_id)

    def set_risk_level(self, level: str) -> str:
        """Set the user's risk tolerance to conservative, moderate, or aggressive (the levels on the profile). Returns the level that was saved."""
        saved = self._book().set_risk_level(self.account_id, level)
        self._forget_saved_profile()
        return saved

    def calculate_total_gain(self, start: date, end: date) -> Decimal:
        """Realized profit on sells from `start` through `end`, in dollars. Deposits and shares still held are not counted."""
        return self._book().calculate_total_gain(self.account_id, start, end)

    def number_trades(self, start: date, end: date) -> int:
        """How many buys and sells were booked from `start` through `end`."""
        return self._book().number_trades(self.account_id, start, end)

    def number_approved_recommendations(self, start: date, end: date) -> int:
        """Recommendations the desk or the user approved (`yes_auto`, `yes_user`) whose decision date is in the window. A pending answer is not counted."""
        return self._book().number_approved_recommendations(self.account_id, start, end)

    def number_denied_recommendations(self, start: date, end: date) -> int:
        """Recommendations refused in the window: the system, the user, an expired approval, or a lack of funds."""
        return self._book().number_denied_recommendations(self.account_id, start, end)

    def account_balance(self) -> Decimal:
        """Settled cash, in dollars: ALL of it, what the account holds (the cash a waiting order has set aside is still in it). What may be SPENT is `available_cash` (design choices entry 140)."""
        return self._book().account_balance(self.account_id)

    def available_cash(self) -> Decimal:
        """The cash that may be spent: the settled cash less what the account's waiting orders have set aside (design choices entry 140: the book's one door, `AccountStore.available_cash`)."""
        return self._book().available_cash(self.account_id)

    def open_orders(self, statuses: tuple[str, ...] = ("open",)) -> list:
        """The account's waiting limit orders (`OpenOrder`s; the open ones by default): each holds cash (a purchase) or shares (a sale) that no other order may use (design choices entry 140)."""
        return self._book().open_orders(self.account_id, statuses)

    def cancel_open_order(self, order_id: int, when: datetime | None = None):
        """The user cancels a waiting order: what it set aside is available again, in one transaction. Returns the order. ValueError (nothing changed) when there is none of that number or it is not open."""
        closed = self._book().cancel_open_order(self.account_id, order_id, when)
        orders.audit({"event": "cancelled", "text": f"order #{closed.order_id} cancelled by the user: {orders.order_words(closed)}", "order": closed.as_dict()})
        return closed

    # ---- what the user is shown (the user talks to the Account Manager only)

    @property
    def guard(self) -> Guard:
        if self._guard is None:
            self._guard = Guard.code_only()
        return self._guard

    @guard.setter
    def guard(self, value: Guard | None) -> None:
        self._guard = value

    def tell_user(self, *lines: str) -> None:
        """Say these lines to the user on the Account Manager's screen, if it has one: each line goes through the security layer first (`guard.screen_message`, entry 116), which redacts a secret and another account's number in place and shows the
        rest as it is. Never raises: a screen that fails does not stop a trade."""
        if self.screen is None:
            return
        try:
            for line in lines:
                self.screen(self.guard.screen_message(line, account_id=self.account_id).text)
        except Exception:
            pass

    def show_portfolio(self, title: str = "the account as it stands now") -> None:
        """Show the user the account as it is now: the cash and each holding (`holding_lines`, the listing `format_account` prints), read from the book or, when no account is saved, from the run's working portfolio."""
        if self.screen is None:
            return
        try:
            portfolio = self.current_portfolio() if self._session else self.list_portfolio()
        except Exception:   # it could not be read: the user is told only what is known
            self.tell_user(f"      ({title}: the account could not be read just now)")
            return
        cash = [f"        cash {portfolio.cash_avail:,.2f}"] if not portfolio.reserved_cash else [f"        cash {portfolio.cash_total:,.2f}", f"        set aside for waiting orders {portfolio.reserved_cash:,.2f}", f"        available {portfolio.cash_avail:,.2f}"]
        self.tell_user(f"      {title}:", *cash, *[f"        {line}" for line in holding_lines(portfolio)])

    def record_trade(self, symbol: str, side: str, shares: Decimal, price: Decimal, traded_on: date, *, name: str | None = None, source: str = "bot", record_id: str | None = None) -> Decimal:
        """Book a filled trade in the ledger and return that trade's realized gain. This does not send an order and does not decide anything: the Trader's `PaperBroker` simulates the fill; `notify_trades` calls this for each fill it is given when it is asked to `book`, and `record_fill`
        books through the store's `book_trade` (the same transaction, with the reads around it: entry 82) for each fill the Trader reports when it is asked to `book` (source "bot", the recommendation's `record_id`). Nothing else books a trade by itself: a fill the Trader made without `book` lives on the work order only."""
        return self._book().record_trade(self.account_id, symbol, side, shares, price, traded_on, name=name, source=source, record_id=record_id)

    @property
    def clerk(self):
        if self._clerk is None:
            from prometheus.agents.data_clerk.data_clerk_agent import DataClerk  # imported here: the Data Clerk needs the toolbox's dependencies and builds its own model
            self._clerk = DataClerk()
        return self._clerk

    # ---- raising funds for a purchase the cash does not cover (entry 142; funding.py)

    def _direct_costs(self, order: WorkOrder, prices: dict[str, Decimal], budget: Decimal | None) -> list[tuple[dict, Decimal]] | None:
        """[(the user's order, what it costs at most)] for every direct BUY that is placed now: typed shares at the price, the dollars the user wrote for the stock, and ONE total for the stocks named with no number (the budget). None when
        a stock has no number and there is no budget ("as much as I can": the cash is its amount, so nothing can be short of it). Orders that wait or are not placed at this price (`direct.not_now`) and sales are left out."""
        from prometheus.agents.account_manager import direct

        rows, bare = [], []
        for entry in order.direct_orders:
            if not isinstance(entry, dict) or entry.get("side") != "buy":
                continue
            price = prices.get(entry.get("symbol"))
            if not price or direct.not_now(entry, price):
                continue
            if entry.get("shares") is not None:
                rows.append((entry, trade_amount(sh.quantity(entry["shares"]), price)))
            elif entry.get("dollars") is not None:
                rows.append((entry, money_door(entry["dollars"])))
            else:
                bare.append(entry)
        if bare:
            if budget is None:
                return None
            rows += [(bare[0], money_door(budget)), *((entry, Decimal("0.00")) for entry in bare[1:])]
        return rows

    def _funding_map(self, order: WorkOrder, prices: dict[str, Decimal], budget: Decimal | None, shortfall: Decimal, by: str) -> dict[str, dict]:
        """{record_id: what its decision carries (`funding.decision_funding`)} for every direct buy that is part of the run's funding (R7): all of them, the ones the cash alone would cover included, because the first one placed uses the cash."""
        return {entry["record_id"]: funding.decision_funding(shortfall, by) for entry, _ in (self._direct_costs(order, prices, budget) or [])}

    def _own_sales(self, order: WorkOrder, portfolio: Portfolio, prices: dict[str, Decimal], now) -> tuple[list[dict], Portfolio]:
        """([{symbol, shares, price, amount}], the portfolio once they are filled) for the user's OWN SALES of this request that are placed now (design choices entry 143, S1 to S3: "Sell 3 MCD and buy 2 ANT, raise funds if needed"). A sale
        is placed before any purchase (the Trader's rule, `PaperBroker.place`), so its proceeds are cash by the time a purchase is approved: they count toward what the purchases need, and the shares it sells are not shares RaiseFunds may
        sell again. Only a sale of typed shares, priced, that is not waiting or under its floor (`direct.not_now`) and that the account holds and may sell today counts; the amount is `trade_amount(shares, price)`, an ESTIMATE at the quote: the
        approval of each purchase still reads the cash the book shows then, so nothing is ever spent that is not there. No sale: ([], the portfolio given). Never raises."""
        from prometheus.agents.account_manager import direct

        today, sales, after = eastern_date(now), [], portfolio
        for entry in order.direct_orders:
            if not isinstance(entry, dict) or entry.get("side") != "sell" or entry.get("shares") is None:
                continue
            price = prices.get(entry.get("symbol"))
            if not price or direct.not_now(entry, price):
                continue
            try:
                count = sh.quantity(entry["shares"])
                if count > after.sellable_shares(entry["symbol"], today):
                    continue
                after = apply_fill(after, entry["symbol"], "sell", count, price, today)
            except ValueError:
                continue
            sales.append({"symbol": entry["symbol"], "shares": count, "price": price, "amount": trade_amount(count, price)})
        return sales, after

    def _waiting_funding(self, order: WorkOrder, portfolio: Portfolio, prices: dict[str, Decimal], now, raise_funds: bool, request_text: str) -> tuple[dict[str, str], dict[str, str]]:
        """({record_id: how the user permitted it}, {record_id: why a purchase is not funded}) for the direct PURCHASES of this request that WAIT for their limit and that the free cash does not cover (design choices entry 143, W-F1 to W-F5;
        `funding.py`). Nothing is sold now: a waiting order is paid for by selling holdings only WHEN ITS PRICE IS REACHED (`fill_open_order`), so the buy and its sales happen together, never the sales alone for a purchase that may never come.
        Nobody is there to approve a plan at that moment, so only the user's own WORDS ("raise funds if needed") or the setting ALWAYS_RAISE_FUNDS permit it (`funding.permission`: "words" or "setting"); in Ask mode without them the order is
        refused for want of cash, as before, and says what to add. Before it waits, the arithmetic says whether the holdings that may be sold could pay for it at its limit (`_holdings_could_raise`, R8): if not it is refused and says what
        they could raise. The cash that is free is set aside; the rest is the shortfall at the limit. Never raises."""
        from prometheus.agents.account_manager import orders

        by, why = {}, {}
        for entry in order.direct_orders:
            price = prices.get(entry.get("symbol")) if isinstance(entry, dict) else None
            if not isinstance(entry, dict) or entry.get("side") != "buy" or not price or not orders.waits(entry, price):
                continue
            limit = orders.limit_of(entry)
            shares = sh.quantity(entry["shares"]) if entry.get("shares") is not None and entry.get("how_many") in (None, "typed") else None
            cost = trade_amount(shares, limit) if shares is not None else (money_door(entry["dollars"]) if entry.get("dollars") is not None else None)
            if limit is None or cost is None or cost <= portfolio.cash_avail:
                continue
            short = cost - portfolio.cash_avail
            permitted = funding.permission(raise_funds, request_text)
            if permitted is None:
                why[entry["record_id"]] = f"nothing is sold unless you say so, and a waiting order is paid for by selling only on your own words (the sale happens when the price is reached, with nobody there to approve it): {funding.NOT_SAID}"
                continue
            feasible, note = self._holdings_could_raise(order, portfolio, cost, now)
            if not feasible:
                why[entry["record_id"]] = f"raising funds could not pay for it: ${short:,.2f} is short of the cash ${portfolio.cash_avail:,.2f} at your limit, and {note}"
                self.tell_user(f"      {why[entry['record_id']]}")
                continue
            by[entry["record_id"]] = permitted
        return by, why

    def _direct_funding(self, order: WorkOrder, portfolio: Portfolio, prices: dict[str, Decimal], now, mode: str | None, approver, raise_funds: bool, request_text: str,
                        budget: Decimal | None, sales: list[dict] | None = None) -> tuple[str | None, Decimal, dict[str, str]]:
        """(how the user permitted selling holdings or None, the shortfall, {record_id: why a purchase of typed shares is not funded}) for the direct buys of this request (see `decide_direct`). The shortfall is what the buys cost at most less the
        available cash (`cash_avail`, the book's door, entry 140) and less what the user's own sales of the same request bring (entry 143, S1: `portfolio` is then the account once those sales are filled, and `sales` says them); no shortfall, no question. Before any question the arithmetic says whether the holdings that may be sold could pay for it at all (R8: `sizing.funds_check`); when they
        could not, nothing is asked and nothing is sold. Never raises: a price that cannot be had leaves the purchase unfunded."""
        rows = self._direct_costs(order, prices, budget)
        if not rows:
            return None, Decimal("0.00"), {}
        need = sum((cost for _, cost in rows), Decimal("0.00"))
        shortfall = need - portfolio.cash_avail
        if shortfall <= 0:
            return None, Decimal("0.00"), {}
        mode = mode or self.profile.trade_mode
        typed = {entry["record_id"] for entry, _ in rows if entry.get("shares") is not None}
        buys = [{"symbol": entry["symbol"], "shares": sh.quantity(entry["shares"]) if entry.get("shares") is not None else sh.buys(cost, prices[entry["symbol"]]), "price": prices[entry["symbol"]], "cost": cost} for entry, cost in rows if cost > 0]
        refuse = lambda why: (None, shortfall, {record: why for record in typed})
        sales = sales or []
        cash = funding.cash_words(portfolio.cash_avail, sales)      # "cash $60.14", or with the user's own sale of the request: "cash $60.14 plus about $348.61 from your own sale of 1 DPZ"
        feasible, note = self._holdings_could_raise(order, portfolio, need, now)
        if not feasible:
            line = f"raising funds could not pay for it: ${shortfall:,.2f} is short of the {cash}, and {note}"
            self.tell_user(f"      {line}")
            return refuse(line)
        by = funding.permission(raise_funds, request_text)
        if by is None and mode == "ask" and approver is not None and getattr(approver, "available", lambda: True)():
            answer = self._ask_raise_funds(order, approver, funding.ask_card(portfolio.cash_avail - sum((x["amount"] for x in sales), Decimal("0.00")), buys, shortfall, sales))
            if answer is True:
                by = "asked"
            else:
                why = "you did not want holdings sold to pay for it" if answer is False else "nobody answered the question whether to raise the money by selling holdings, and no answer is a no"
                self.tell_user(f"      Nothing is sold: {why}.")
                return refuse(why + ": the order is placed as you typed it or not at all")
        if by is None:
            why = f"nothing is sold in Auto mode unless you say so: {funding.NOT_SAID}" if mode == "auto" else f"nobody was there to ask whether to raise the money by selling holdings: {funding.NOT_SAID}"
            self.tell_user(f"      The {cash} is ${shortfall:,.2f} short of what you asked to buy, and {why}.")
            return refuse(why)
        self.tell_user(f"      The {cash} is ${shortfall:,.2f} short of what you asked to buy: {funding.how_words(by)}, so I will raise the money by selling some of your holdings first (RaiseFunds), and then buy.")
        return by, shortfall, {}

    def _ask_raise_funds(self, order: WorkOrder, approver, card: dict) -> bool | None:
        """The ONE question of Ask mode (R3): the card of the kind `raise_funds`. True for a yes, False for a no, None for no answer (or a prompt that failed: the user has not said yes)."""
        try:
            answer = approver(card)
        except Exception as error:
            order.errors = [*order.errors, f"the user could not be asked whether to raise funds ({type(error).__name__})"]
            return None
        return answer if isinstance(answer, bool) else None

    def _holdings_could_raise(self, order: WorkOrder, portfolio: Portfolio, need: Decimal, now) -> tuple[bool, str]:
        """(whether selling the holdings that may be sold could raise `need` with the cash, a sentence): arithmetic on the holdings' prices (one quote order to the Data Clerk: `_held_prices`) and `sizing.funds_check`, no model. R8: when it is
        not, the sentence says what they could raise."""
        prices = self._held_prices(order, portfolio, now, True)
        return sizing.funds_check(need, portfolio, prices, eastern_date(now))

    @staticmethod
    def _after_decided_sales(order: WorkOrder, portfolio: Portfolio, now) -> Portfolio:
        """The account once the user's own sales that the decisions of this order permit (`origin` direct, a sale, `yes_user` or `yes_auto`) are filled at their decision's price (entry 143, S2): the same arithmetic the Account Manager
        made when it decided (`_own_sales`), read back from the decisions. A sale the account cannot make is left out (the Trader's handshake refuses it). No such sale: the portfolio given."""
        today, after = eastern_date(now), portfolio
        for d in order.trade_decisions:
            if not isinstance(d, dict) or d.get("origin") != DIRECT or d.get("action") != "reduce" or d.get("will_trade") not in ("yes_auto", "yes_user") or not d.get("price"):
                continue
            try:
                after = apply_fill(after, d["symbol"], "sell", sh.quantity(d["shares"]), price_door(d["price"]), today)
            except ValueError:
                continue
        return after

    @traced("account_manager.fund_decisions")
    def fund_decisions(self, order: WorkOrder, portfolio: Portfolio, now=None) -> dict | None:
        """RaiseFunds for the purchases the decisions say are to be paid for by selling (entry 142, R6 and R10): the decisions that carry `funding` and may trade (`funding.funded_decisions`). ONE plan raises the WHOLE shortfall of all of them
        (the first is RaiseFunds' target and the others its `also`: no stock being bought is ever sold), for the shares as decided (whole shares: a fraction is raised for the next whole one). Returns the answer for the Trader's funding plan
        (`perform_trades(..., funding=)`): RaiseFunds' own, plus `basket` (the symbols), `plan_id` (written on the decisions), `by` (how the user permitted it) and `ask_user` (False only in Auto mode, where the user's own words or the setting are the
        approval of the sales; Ask mode always shows the plan and asks). None when no decision needs it. When RaiseFunds cannot pay for it (R8: no plan, no price, the search failed) every such decision becomes `no_funds`, with the shortfall and
        what the holdings could raise, the user is told and the answer says why; nothing is sold. Never raises."""
        needing = funding.funded_decisions(order)
        if not needing:
            return None
        now = now or trading_now()
        portfolio = self._after_decided_sales(order, portfolio, now)      # entry 143, S2: the user's own sales of this request are placed first, so RaiseFunds works from the account once they are filled: their proceeds are cash, and their shares are not for sale again
        by, mode = needing[0]["funding"]["by"], needing[0].get("mode") or self.profile.trade_mode
        buys = [(d["symbol"], sh.ceil(sh.quantity(d["shares"]))) for d in needing]
        try:
            result = self.raise_funds(portfolio, buys[0][0], buys[0][1], eastern_date(now), up_to=False, also=buys[1:] or None)
        except Exception as error:
            result = {"status": "error", "errors": [f"RaiseFunds failed ({type(error).__name__})"], "quotes": {}, "run": {}}
        answer = {**result, "basket": [symbol for symbol, _ in buys], "by": by, "ask_user": not (mode == "auto" and by in ("words", "setting"))}
        if result.get("status") in ("plan", "funded"):
            answer["plan_id"] = plan_id = f"{order.request_id}-funds"
            for d in needing:
                d["funding"]["plan_id"] = plan_id if result["status"] == "plan" else None
            return answer
        need = result.get("need") or needing[0]["funding"]["shortfall"]
        why = "; ".join(map(str, result.get("errors") or [])) or str(result.get("stop") or result.get("status"))
        held = _prices_of(dict(result.get("quotes") or {}), portfolio.symbols)
        could = sizing.funds_check(max(need, Decimal("0.01")) + portfolio.cash_avail, portfolio, held, eastern_date(now))[1] if held else "no holding of the account could be priced"
        line = f"RaiseFunds could not raise the ${need:,.2f} that is short ({why}); {could}"
        for d in needing:
            d.update(will_trade="no_funds", funds_avail=False, reason=" ".join(line.split())[:400] + ": the purchase is placed as you typed it or not at all, and nothing is sold")
        self.tell_user(f"      {line}: nothing is sold and nothing is bought.")
        self._save_decisions(order)
        return {**answer, "plan_id": None}

    # ---- the Data Clerk: prices

    def request_prices(self, symbols: list[str]):
        """Ask the Data Clerk for the current price of each symbol (the design's `DataClerk.request_data`, for the `fetch_quote` action). Returns the work order, with `data.quotes`."""
        order = new_work_order(f"Current prices of {', '.join(symbols)}", requester=REQUESTER, action_type="fetch_quote", subject=list(symbols), live=True)
        return self.clerk.request_quotes(order)

    def request_bars(self, symbols: list[str], start: date, end: date):
        """Ask the Data Clerk for the stored daily bars of each symbol from `start` to `end`, both included (a `fetch_bars` order, answered by `DataClerk.request_bars`: no model, no fetch). Beside `request_prices`: the Trader's
        backtest prices come this way, never from the Trader asking the Data Clerk itself. Returns the work order, with `data.bars` ({symbol: [bar, ...]}, oldest first); a symbol with no bar in the range is an empty list.
        Raises ValueError when `start` or `end` is not a date (a programming mistake); anything the Data Clerk refuses is on the order, which is then `data_failed`."""
        if type(start) is not date or type(end) is not date:
            raise ValueError("start and end must be dates")
        order = new_work_order(f"Daily bars of {', '.join(symbols)} from {start.isoformat()} to {end.isoformat()}", requester=REQUESTER, action_type="fetch_bars", subject=list(symbols),
                               bars={"start": start.isoformat(), "end": end.isoformat()})
        return self.clerk.request_bars(order)

    # ---- the Trader (Trader v1, design/trader_v1_final_spec.md)

    @traced("account_manager.trading_portfolio")
    def trading_portfolio(self, portfolio: Portfolio | None = None, book: bool = False) -> Portfolio:
        """Open a trading run and return the portfolio the Trader starts from: the saved account when this manager is connected (`list_portfolio`: the book is the truth), else the one given (the desk's JSON file).
        Raises ValueError when there is neither. This is also where the Account Manager takes the funds into its own hands for the per-trade handshake (design choices entry 81): with a saved account and `book` true every
        `approve_trade` reads the book and every `record_fill` books in it; otherwise the portfolio returned (a snapshot of the saved account when `book` is false, the one given when nothing is connected) is stored as
        the Account Manager's WORKING portfolio, which `approve_trade` reads and `record_fill` changes with `apply_fill`, booking nothing. `book` true with no saved account is the same, and the first `record_fill` says so
        (`Booking.note`). A new call starts a new run: whatever the last one left is forgotten. The portfolio given is never changed."""
        self._run_fills = {}
        if self.account_id is not None:
            saved = self._portfolio()
            self._session, self._booking, self._book_unavailable, self._working = True, bool(book), False, None if book else saved
            return saved
        if portfolio is None:
            raise ValueError("there is no portfolio to trade: connect an account (store= and account_id=) or give one")
        self._session, self._booking, self._book_unavailable, self._working = True, False, bool(book), portfolio
        return portfolio

    def spent_of_pot(self, order: WorkOrder, pot: Decimal) -> Decimal:
        """What the purchases of this work order that carry THIS pot have cost so far, in dollars (design choices entry 111, M1): the filled purchases on `order.trades` (the fills of an earlier run, which a resumed run counts)
        and the fills `record_fill` was told in this run, each once by its trade id, never from memory alone. A purchase carries the pot when its decision's `size.pot` is that amount. Sales and funding sales are not
        purchases and cost nothing against it."""
        found: dict[str, dict] = {t["trade_id"]: t for t in order.trades if isinstance(t, dict) and isinstance(t.get("trade_id"), str)}
        found.update(self._run_fills)
        total = Decimal(0)
        for trade in found.values():
            fill = trade.get("fill")
            if trade.get("status") != "filled" or trade.get("side") != "buy" or trade.get("kind") == "funding" or not isinstance(fill, dict):
                continue
            decision = decision_of(order, trade.get("record_id")) if isinstance(trade.get("record_id"), str) else None
            if decision is not None and decision_pot(decision) is not None and decision_pot(decision) == pot:
                total += digits(fill.get("amount") or 0)      # a Decimal from this run, or the int or float that `json` gave for a fill read back from a saved order
        return total

    def current_portfolio(self) -> Portfolio:
        """The account as it stands NOW in the open trading run: the book (`list_portfolio`) when the run books, else the working portfolio. What the Trader reports as the account after a trade or after the run: it
        is the Account Manager's figure, never the Trader's. Raises ValueError when no run is open (call `trading_portfolio` first); a database failure raises as `list_portfolio` does."""
        if not self._session:
            raise ValueError("no trading run is open: call trading_portfolio first")
        return self._portfolio() if self._booking else self._working

    def _portfolio(self) -> Portfolio:
        """The saved account as THIS run reads it: the book's `list_portfolio` (the available cash, the whole holding and what waiting orders hold), and, while this manager is FILLING a waiting order (`fill_open_order`, design choices entry 140), with that
        order's own hold left out: it is available to its own trade and to nothing else."""
        if self._filling is None:
            return self.list_portfolio()
        return self._book().list_portfolio(self.account_id, release=self._filling.order_id)

    @traced("account_manager.apply_earlier_fills")
    def apply_earlier_fills(self, trades: list[dict]) -> list[str]:
        """Put into the working portfolio the fills of an EARLIER run that were never booked (the `filled` trades of `order.trades` with `booked` false: they live only on the order), sales first, with `apply_fill`
        and nothing booked, so that a second run of the same order sees the cash the first one spent and a `filled` trade stays final (rule T3). Call it after `trading_portfolio` and before the first approval. With a
        run that books, the book already has every fill it was told, so nothing is applied: a fill that an earlier run left unbooked is named as a problem (it is not in the account book). Returns the problems (one
        sentence each, starting with the trade id); a fill that cannot be applied is a problem and the others still go on. Raises ValueError when no run is open."""
        if not self._session:
            raise ValueError("no trading run is open: call trading_portfolio first")
        problems = []
        for trade in handshake.unbooked_fills(trades):
            trade_id, fill = trade.get("trade_id") or "a trade", trade["fill"]
            if self._booking:
                problems.append(f"{trade_id}: filled in an earlier run that did not book it, so it is not in the account book")
                continue
            when = iso_date(fill.get("date"))
            try:
                if when is None:
                    raise ValueError(f"its fill date {fill.get('date')!r} is not a date")
                self._working = apply_fill(self._working, trade.get("symbol"), trade.get("side"), sh.quantity(fill.get("shares")), price_door(fill.get("price")), when)      # the numbers of a fill read back from a saved order go through their doors
            except ValueError as error:
                problems.append(f"{trade_id}: its earlier fill was not applied to the portfolio: {error}")
        return problems

    @traced("account_manager.approve_trade")
    def approve_trade(self, trade: dict, price: Decimal, *, order: WorkOrder, now: datetime, on: date, commission: Decimal = Decimal(0)) -> handshake.Approval:
        """The first half of the per-trade handshake (design choices entry 81): may the Trader make THIS trade now, and for how many shares? Reads the funds and the holdings as they are at this moment (the book when the
        run books, else the working portfolio) and answers an `Approval`: for a purchase the decision's shares cut to what the available cash pays for at `price` (`affordable_shares`: whole shares for a whole order, a fractional
        quantity for a fractional one; the account book's rounding and the Trader's `commission` setting), for a sale the shares held and sellable on `on` (T+1) up to the decision's. It says no (`code`: `insufficient_cash`, `not_held`, `not_sellable_today`,
        `not_permitted`, `approval_expired`, `account_refused`) when nothing can be traded, when the decision no longer permits it (`work_order.verify_trade` over `order` at `now`), when the account is not active or
        when the account cannot be read. `trade` is the Trader's record of the trade as a dict (`order.trades` shape: `shares` is what is asked, `shares_approved` what the decision allowed); `price` the price it would
        fill at. Nothing is reserved: the answer is about the funds of this moment, and the next trade meets whatever `record_fill` leaves. Never raises for a data or database failure: it does not approve, and the
        sentence names the kind of the error and never a connection detail.

        A RUN NEVER SPENDS MORE THAN ITS MONEY (design choices entry 111). A purchase whose decision carries a pot (`size.pot`) is approved for no more than what is left of it: the pot less what the purchases of this work order that carry
        it have cost (`spent_of_pot`: the fills on `order.trades` and the fills `record_fill` was told in this run). A purchase whose decision is an ORDER IN DOLLARS (`dollars`) is approved for the shares those dollars pay for AT `price`
        (`shares.buys`): fewer than the decision's estimate when the price is higher, more when it is lower. A purchase with no pot (the shares of a direct order the user typed, a RaiseFunds purchase) is held by the cash only."""
        return self._approve(trade, price, order, now, on, commission, Decimal(0))

    @traced("account_manager.preview_trade")
    def preview_trade(self, trade: dict, price: Decimal, *, order: WorkOrder, now: datetime, on: date, commission: Decimal = Decimal(0), expected_proceeds: Decimal = Decimal(0)) -> handshake.Approval:
        """A QUESTION, not an approval (nothing is placed, reserved or changed): what `approve_trade` would answer if `expected_proceeds` more dollars were available. The Trader asks it twice around a funded purchase
        (spec section 7), so that it never works out the funds itself: with 0 to know whether the cash covers the whole purchase (the user is only asked to approve the funding sales when it does not), and with the
        estimated proceeds of the funding sales to know whether they would pay for even one share (when they would not, nothing is sold). The answer of the real approval comes later, from the cash the book shows then."""
        return self._approve(trade, price, order, now, on, commission, expected_proceeds)

    def _approve(self, trade: dict, price: Decimal, order: WorkOrder, now: datetime, on: date, commission: Decimal, expected_proceeds: Decimal) -> handshake.Approval:
        try:
            if not self._session:
                raise ValueError("no trading run is open: call trading_portfolio first")
            reasons = verify_trade(order, trade.get("record_id"), now)
            if reasons:
                return handshake.refusal(handshake.permission_code(reasons), "; ".join(reasons))
            if self._booking:
                status = self._book().get_user(self.account_id).status
                if status != "active":
                    return handshake.refusal("account_refused", f"the account is {status}: it cannot take a trade")
            decision = decision_of(order, trade.get("record_id"))
            pot = decision_pot(decision) if decision is not None and trade.get("side") == "buy" and trade.get("kind") != "funding" else None
            spendable = None if pot is None else max(Decimal("0.00"), pot - self.spent_of_pot(order, pot))      # what the run may still spend of its money, from the fills on the work order and in this run (M1)
            dollars = decision_dollars(decision) if decision is not None and trade.get("side") == "buy" and trade.get("kind") != "funding" else None      # an ORDER IN DOLLARS (entry 111, M2): from the decision, which is the Account Manager's own, never from the Trader's record of it
            limit = decision_max_price(decision) if decision is not None and trade.get("side") == "buy" and trade.get("kind") != "funding" else None      # the user's price limit (entry 138): from the decision, the Account Manager's own, never from the Trader's record
            floor = decision_min_price(decision) if decision is not None and trade.get("kind") != "funding" else None      # the user's price floor (entry 140): from the decision too, for a purchase and for a sale
            answer = handshake.decide_approval(self.current_portfolio(), trade, price, on, commission, expected_proceeds, spendable, pot, dollars, limit, min_price=floor)
            if answer.approved and trade.get("kind") != "funding" and decision is not None and decision.get("origin") == DIRECT and dollars is None and answer.shares < sh.quantity(decision.get("shares", 0)):
                # the user named this trade (entry 87): it is placed as typed or not at all, never cut to the cash or to the shares sellable today
                return handshake.refusal("insufficient_cash" if trade.get("side") == "buy" else "not_sellable_today", f"the order is for {sh.shown(decision.get('shares'))} shares and is placed as you typed it or not at all, and the account can take {sh.text(answer.shares)}: {answer.reason}",
                                         answer.cash_avail)
            return answer
        except AssertionError:   # a programming mistake or a test's guard: not hidden
            raise
        except Exception as error:   # a database failure, or anything else: the trade is not approved, and the sentence names the kind only
            sentence = f"the Account Manager could not read the account ({type(error).__name__}): the trade is not approved"
            self.tell_user("", f"  !! TRADE NOT DONE: {trade.get('side')} of {sh.shown(trade.get('shares'))} {trade.get('symbol')}: {sentence}")
            return handshake.refusal("account_refused", sentence)

    @traced("account_manager.record_fill")
    def record_fill(self, trade: dict) -> handshake.Booking:
        """The second half of the per-trade handshake (design choices entries 81 and 82): the Trader reports ONE trade as filled, at once, and waits for the answer before it asks for the next trade.
        A run that BOOKS (a saved account): the account is SELECTed, the fill is COMMITTED in ONE Postgres transaction (`AccountStore.book_trade`: the cash and the position move together, or neither, and the
        transaction re-reads the account before it commits and rolls back on a difference), and the account is SELECTed AGAIN on a new connection to confirm that the trade is in the book and that the cash and the
        shares are different from before (`_book_fill`). Without booking the fill is applied to the working portfolio with `apply_fill` and nothing is booked. Returns a `Booking`: `booked` (with `verified`),
        `applied`, `refused` (nothing was committed: the book or the portfolio said no, with its own sentence, or a failed connection left the account unchanged) or `unconfirmed` (nobody can settle it from here:
        it is NEVER assumed that the trade was booked). A trade that is not done is shown to the user at once, by the Account Manager, with the account as it stands; it is never retried, and the Trader goes on to the
        next trade. Never raises for a data or database failure. `trade` is the Trader's record of the filled trade as a dict (`order.trades` shape)."""
        try:
            symbol, side, shares, price, when, record_id = handshake.fill_parts(trade)
            if not self._session:
                raise ValueError("no trading run is open: call trading_portfolio first")
        except ValueError as error:   # nothing was tried: nothing changed
            return self._not_done("refused", f"the fill of {trade.get('trade_id') if isinstance(trade, dict) else 'a trade'}", " ".join(str(error).split()))
        if not self._booking:
            try:
                self._working = apply_fill(self._working, symbol, side, shares, price, when)
            except ValueError as error:
                return self._not_done("refused", f"the {side} of {sh.shown(shares)} {symbol}", " ".join(str(error).split()), cash_avail=self._working.cash_avail)
            note, self._book_unavailable = (handshake.NO_STORE_NOTE if self._book_unavailable else ""), False   # said once, in the first fill
            self._remember_fill(trade)
            return handshake.Booking(status="applied", cash_avail=self._working.cash_avail, note=note)
        booking = self._book_fill(symbol, side, shares, price, when, record_id)
        if booking.status in ("booked", "unconfirmed"):      # an unconfirmed one may be in the book: the pot is a ceiling, so it counts as spent
            self._remember_fill(trade)
        return booking

    def _remember_fill(self, trade: dict) -> None:
        """Keep the fill the Trader just reported, by its trade id, so that the next approval knows what the run has spent of its pot (`spent_of_pot`)."""
        self._run_fills[str(trade.get("trade_id") or f"fill-{len(self._run_fills) + 1}")] = {**trade, "fill": dict(trade["fill"])}

    def _book_fill(self, symbol: str, side: str, shares: Decimal, price: Decimal, when: date, record_id: str | None) -> handshake.Booking:
        """Book ONE fill and check it by reading the account twice (design choices entry 82), the Account Manager's own steps:
          1. SELECT the account as it is now (`list_portfolio`: the cash and the holdings).
          2. COMMIT the trade: `AccountStore.book_trade`, one transaction that locks the account row, moves the cash and the position together, writes the trade row and, before it commits, reads the account again
             inside it; a result that is not the one the trade must make rolls the whole transaction back.
          3. SELECT again, on a NEW connection, after the commit: the trade row must be in the book as it was placed, and the cash and the shares of the symbol must be different from step 1.
        Only then is the answer `booked` (with `verified`), and the Trader goes on to its next trade. An error at any step makes the trade NOT DONE: the answer is `refused` (nothing was committed: the book said no,
        or a failed connection left the account unchanged when it was read again) or `unconfirmed` (nobody can settle it from here), the user is shown the error and the account as it stands, and the trade is never
        retried; the Trader goes on to the next one, whose approval reads the account again."""
        label = f"the {side} of {sh.shown(shares)} {symbol}"
        try:
            before = self.list_portfolio()                                   # 1
        except Exception as error:
            return self._not_done("refused", label, f"the account could not be read before the booking ({type(error).__name__}): nothing was booked")
        try:
            booked = self._book().book_trade(self.account_id, symbol, side, sh.as_decimal(shares), price, when, source="bot", record_id=record_id,
                                             **({"order_id": self._filling.order_id} if self._filling is not None and record_id == self._filling.record_id and side == self._filling.side and symbol == self._filling.symbol else {}))       # 2 (a waiting order's fill marks the order filled in this same transaction, entry 140; a funding sale of that fill carries the same record_id but is a sale of another stock, a booking of its own: entry 143)
        except ValueError as error:   # the book's own refusal (the cash does not cover it, the shares are not held, the account is not active) or its check before the commit: rolled back, nothing was committed
            return self._not_done("refused", label, " ".join(str(error).split()), cash_before=before.cash_avail)
        except Exception as error:    # a database or network failure while the transaction was open: the account is read again to settle it, never guessed
            return self._after_failure(before, symbol, side, shares, label, error)
        try:
            after = self.list_portfolio()                                    # 3 (a new connection: what the commit made durable)
            row = self._book().read_booked(self.account_id, booked.trade_row)
        except Exception as error:    # committed (book_trade returned), and only the second read failed: the trade stands, and the user is told it was not confirmed
            note = f"{label} is booked, but the account could not be read again to confirm it ({type(error).__name__})"
            self.tell_user(f"      !! {note}")
            return handshake.Booking(status="booked", cash_avail=booked.cash_after, cash_before=before.cash_avail, note=note)
        problems = handshake.check_booked(before, after, row, symbol, side, shares)
        if problems:                  # the book said yes and the account, read again, does not show it: it cannot be un-committed from here, so it is reported as it is and checked by a person
            return self._not_done("unconfirmed", label, "the booking was committed but the account, read again, does not show it: " + "; ".join(problems) + ": the account book has to be checked",
                                  cash_before=before.cash_avail, cash_avail=after.cash_avail)
        return handshake.Booking(status="booked", cash_avail=after.cash_avail, cash_before=before.cash_avail, verified=True)

    def _after_failure(self, before: Portfolio, symbol: str, side: str, shares: Decimal, label: str, error: Exception) -> handshake.Booking:
        """The booking raised something that is not the book's own refusal (a lost connection): read the account again. Unchanged means nothing was committed (`refused`); anything else, or no answer, is `unconfirmed`:
        it is never assumed that the trade was, or was not, booked."""
        kind = type(error).__name__
        try:
            now = self.list_portfolio()
        except Exception:
            return self._not_done("unconfirmed", label, f"the booking could not be confirmed ({kind}): the account book may or may not hold this trade, and it could not be read to find out", cash_before=before.cash_avail)
        if handshake.unchanged(before, now, symbol):
            return self._not_done("refused", label, f"the booking failed ({kind}) and the account, read again, is unchanged: nothing was committed", cash_before=before.cash_avail, cash_avail=now.cash_avail)
        return self._not_done("unconfirmed", label, f"the booking could not be confirmed ({kind}): the account book may or may not hold this trade; the account read again is not what it was before",
                              cash_before=before.cash_avail, cash_avail=now.cash_avail)

    def _not_done(self, status: str, label: str, reason: str, **figures) -> handshake.Booking:
        """A trade that is NOT DONE: shown to the user with the account as it stands, and answered as `refused` or `unconfirmed` (the Trader marks it failed and never retries it). The sentence is cut to a sane length and
        never holds a connection detail (the callers give the kind of an error, not its text)."""
        sentence = " ".join(reason.split())[:handshake.SENTENCE]
        self.tell_user("", f"  !! TRADE NOT DONE: {label} was not recorded in the account and will not be tried again.", f"      {sentence}")
        self.show_portfolio()
        return handshake.Booking(status=status, reason=sentence, **figures)

    def perform_trades(self, order: WorkOrder, portfolio: Portfolio, trader, now=None, funding=None, approver=None, book: bool = False) -> WorkOrder:
        """The design's Perform Trade: hand the decided order to the Trader (`trader.execute_trades`) and return it with its trades. The Account Manager only hands over: the permission is in `order.trade_decisions`
        (`decide_trades`), the prices come back through `request_prices` or `request_bars`, and each trade is approved (`approve_trade`) and reported as filled (`record_fill`) one at a time. `funding` is a RaiseFunds result or None, `approver` the same callable
        the decisions use, `book` asks for the fills to be saved in the account book."""
        return trader.execute_trades(order, self, portfolio, now, funding=funding, approver=approver, book=book)

    @traced("account_manager.notify_trades")
    def notify_trades(self, portfolio: Portfolio, trades: list[dict], traded_on: date, book: bool = False) -> tuple[Portfolio, list[str]]:
        """The design's AccountManager.notify, for a caller that reports a LIST of fills to a portfolio it holds (the Trader does not use it any more: since design choices entry 81 it reports each fill AT ONCE with `record_fill`,
        and the Account Manager holds the funds). The trades are the plain dicts of `order.trades` (the dump of the Trader's `Trade` record), and this applies each one whose status is `filled`, in the order
        given, with `apply_fill`, at the fill's own price and shares. It returns the NEW portfolio (the one given is not changed) and the problems. Trades that are not filled are ignored. Give them in the order they were
        filled (a funding sale before the purchase it pays for): a purchase is applied against the cash the sales before it left.

        With `book` true and a connected account, each fill is also booked (`record_trade`: source "bot", the trade's `record_id`, `traded_on` as the date) and `portfolio` is not read from the book again. A fill that cannot be applied
        or booked is a problem and is left out of the portfolio that is returned, so that portfolio is what was booked; the other fills still go through. A problem about one trade starts with its trade id and a colon.
        `book` true without a connected account is ONE problem ("book: ...") and nothing is booked; the fills are still applied to the portfolio in memory. Raises ValueError only for a `traded_on` that is not a date."""
        if type(traded_on) is not date:
            raise ValueError("traded_on must be a date")
        problems: list[str] = []
        booking = book
        if book and (self.store is None or self.account_id is None):
            problems.append("book: the fills were not booked: this Account Manager has no saved account (connect one with store= and account_id=)")
            booking = False
        current = portfolio
        for index, trade in enumerate(trades):
            if not isinstance(trade, dict):
                problems.append(f"trade #{index + 1}: not a record")
                continue
            if trade.get("status") != "filled":
                continue
            trade_id = trade.get("trade_id") or f"#{index + 1}"
            fill = trade.get("fill")
            if not isinstance(fill, dict):
                problems.append(f"{trade_id}: filled, but it has no fill")
                continue
            symbol, side, shares = trade.get("symbol"), trade.get("side"), fill.get("shares")
            try:
                shares, price = sh.quantity(shares), price_door(fill.get("price"))      # the numbers of a fill go through their doors (what is not a number is a problem, named)
                applied = apply_fill(current, symbol, side, shares, price, traded_on)
            except ValueError as error:
                problems.append(f"{trade_id}: {side} {sh.shown(shares)} {symbol} was not applied: {error}")
                continue
            if booking:
                try:
                    self.record_trade(symbol, side, shares, price, traded_on, source="bot", record_id=trade.get("record_id"))
                except ValueError as error:  # the book's own refusal (the cash does not cover it, the shares are not held, the account is suspended): its message is for the user
                    problems.append(f"{trade_id}: {side} {sh.shown(shares)} {symbol} was not booked: {error}")
                    continue
                except Exception as error:  # a database or network error: the kind only, never the connection details
                    problems.append(f"{trade_id}: {side} {sh.shown(shares)} {symbol} was not booked ({type(error).__name__})")
                    continue
            current = applied
        return current, problems

    # ---- the user's own words

    @traced("account_manager.understand_request")
    def understand_request(self, text: str) -> TradeIntent:
        """Read what the user typed ("I want to buy ___ stocks. Buy the 3 most recommended ..., split the money evenly") into a `TradeIntent` (design choices entry 78). The model only maps the words to the form
        (`prompts/account_manager_request.py`); CODE accepts the form only when the user's own text supports every field (`intent.parse_intent`: every ticker, name, number and cue word must be in what was typed),
        so a company, a count or a budget the user never said is refused and the model is asked once more. Nothing here names a sector or a company. Raises ValueError, with a message for the user, when the request is
        empty or too long (no model call), when the model cannot be reached (the kind of the failure, never request details) or when two tries give no form that the words support. `intent_run` says what it cost."""
        typed = check_typed(text)
        usage = self.intent_run = {"model": getattr(self.model, "model_name", None), "calls": 0, "retries": 0, "tokens_in": 0, "tokens_out": 0}
        messages = [SystemMessage(content=request_prompt.SYSTEM), HumanMessage(content=request_prompt.user_message(typed))]
        problem = "no reply"
        for _ in range(2):
            try:
                reply = self.model.invoke(messages)
            except Exception as error:   # a model or network failure: the kind only
                raise ValueError(f"the Account Manager's model could not be reached ({type(error).__name__}): the request cannot be read") from None
            meta = getattr(reply, "usage_metadata", None) or {}
            usage["calls"] += 1
            usage["tokens_in"] += meta.get("input_tokens", 0)
            usage["tokens_out"] += meta.get("output_tokens", 0)
            content = reply.content if isinstance(reply.content, str) else ""
            try:
                return parse_intent(content, typed)
            except NeedsClarification:      # the user has to say more (entry 138): no second try of the model can fix it
                raise
            except ValueError as error:
                problem = str(error)
                usage["retries"] += 1
                messages = [*messages, AIMessage(content=content), HumanMessage(content=request_prompt.RETRY.format(problem=problem))]
        raise ValueError(f"I could not understand the request ({problem}). Say what you want as a theme, as tickers or as company names, and how many you want to buy from.")

    @traced("account_manager.settle_unclear_number")
    def settle_unclear_number(self, intent: TradeIntent, mode: str | None = None, channel=None) -> TradeIntent:
        """A number the user wrote with no $ and no word dollars after a money verb ("spend 1000 on MU") is shares or dollars, and it is never read either way without asking (design choices entry 138, D5: buying 1000 shares
        when $1,000 was meant spends real money nobody intended). In ASK mode, with somebody at the keyboard (`channel`, the same one as the how-much questions: a card with a `question`, answered with the text typed), the user is asked
        once, and once more when the answer names neither; CODE reads the answer (`intent.read_unit_answer`: shares, or a $ or the word dollars) and settles the number (`intent.settle_unit`). In AUTO mode nothing is asked and the run
        stops: `NeedsClarification`, whose message says how to write it. So does Ask mode with nobody to ask, and an answer that never comes clear. An intent with no such number comes back as it is."""
        number = intent.unclear_number
        if number is None:
            return intent
        shown = f"{canonical(number):,f}"
        say = f"You wrote {shown} with no $ and no word dollars, so it could be {shown} shares or ${number:,.2f}"
        how = f"Write the amount with a $ sign or the word dollars (${shown}), or as shares ('{shown} shares of ...'), and send the request again."
        if (mode or self.profile.trade_mode) != "ask" or channel is None or not getattr(channel, "available", lambda: True)():
            raise NeedsClarification(f"{say}, and nothing is bought on a guess (Auto mode asks nothing). {how}")
        for n in (1, 2):
            answer = channel({"question": f"{say}: which did you mean? Answer shares or dollars.", "n": n, "of": 2, "facts": [], "kind": "how_much"})
            if answer is None:
                break
            unit = read_unit_answer(answer, number)
            if unit is not None:
                try:
                    settled = settle_unit(intent, unit)
                except ValueError as error:
                    raise NeedsClarification(f"{say}; you said {unit}, but {error}. {how}") from None
                self.tell_user("", f"  Understood: {shown} {'shares' if unit == 'shares' else 'dollars'}.")
                return settled
        raise NeedsClarification(f"{say}, and the answer did not say which. {how}")

    # ---- the thought generator

    def _generate(self, problem: rf.Problem, node: rf.Node, rejected: list[str]) -> list:
        """One expansion: the model's next steps for this node. One retry if the reply is not the JSON asked for. Raises if the model call fails (the search records it)."""
        messages = [SystemMessage(content=system_prompt(problem.branching, problem.only_green, problem.fill, problem.lean, problem.explore, problem.max_depth, problem.green_first, problem.never_liquidate, problem.whole_plans)), HumanMessage(content=user_message(problem, node, rejected))]
        problem_text = ""
        for _ in range(2):
            reply = self.model.invoke(messages)
            usage = getattr(reply, "usage_metadata", None) or {}
            with self._lock:
                self.run["calls"] += 1
                self.run["tokens_in"] += usage.get("input_tokens", 0)
                self.run["tokens_out"] += usage.get("output_tokens", 0)
            text = reply.content if isinstance(reply.content, str) else ""
            try:
                return parse_thoughts(text)
            except ValueError as error:
                problem_text = str(error)
                with self._lock:
                    self.run["retries"] += 1
                messages = [*messages, AIMessage(content=text), HumanMessage(content=RETRY.format(problem=problem_text))]
        raise ValueError(f"the model's reply was not usable ({problem_text})")

    # ---- RaiseFunds

    @staticmethod
    def _check_purchase(target, shares) -> str:
        """The cleaned target symbol. Raises ValueError, with the reason, if the purchase makes no sense. The number of shares goes through the share door (`shares.quantity`: a bool and a string, "2" too, are refused) and must be whole, at least 1."""
        if not isinstance(target, str):
            raise ValueError("the symbol to buy must be text, like 'MU'")
        target = check_symbol(target.strip().lstrip("$"))
        if shares is not None:
            try:
                count = sh.quantity(shares)
            except ValueError:
                count = None
            if count is None or not sh.whole(count) or count < 1:
                raise ValueError("the number of shares to buy must be a whole number of at least 1 (or left out, to buy as many as possible)")
        return target

    def _resolve(self, explore, lean, green_first, up_to, whole_plans, limits: dict):
        """The arguments that were not given, from the Account Manager's own settings (the defaults of entry 50); `parallel` and `deadline` go into the limits of the search unless they are there."""
        limits = dict(limits)
        limits.setdefault("parallel", self.parallel)
        if self.deadline and "deadline" not in limits:
            limits["deadline"] = self.deadline
        pick = lambda value, own: own if value is None else value
        return pick(explore, self.explore), pick(lean, self.lean), pick(green_first, self.green_first), pick(up_to, self.up_to), pick(whole_plans, self.whole_plans), limits

    def _new_run(self) -> None:
        self.run = {"model": getattr(self.model, "model_name", None), "calls": 0, "retries": 0, "tokens_in": 0, "tokens_out": 0, "error": None, "rate_limited": False}

    def _prices(self, symbols: list[str]) -> tuple[dict, dict | None, list[str]]:
        """(the quotes by symbol, the quote work order as JSON, the reasons it failed: empty when every symbol was quoted)."""
        order = self.request_prices(list(dict.fromkeys(symbols)))
        quotes = dict(order.data.quotes) if order.data else {}
        failed = [] if order.status == "data_ready" else (order.errors or ["the Data Clerk returned no quotes"])
        return quotes, json.loads(order.model_dump_json()), failed

    def _plan(self, portfolio: Portfolio, target: str, shares: Decimal | None, quotes: dict, today: date | None, only_green: bool, fill: str | None = None, explore: bool = False,
              lean: bool = False, green_first: bool = False, up_to: bool = False, never_liquidate: bool = False, auto_depth: bool = True, whole_plans: bool = False,
              also: tuple[tuple[str, Decimal], ...] = (), **limits) -> dict:
        """The tree search for one purchase, with the quotes already in hand. Never raises for a model problem. `limits` are the size of the tree: `max_depth` and `branching` (also told to the
        model), `beam_width` and `max_calls`.

        HOW MANY SHARES (entry 45). `shares` is X. With X and `up_to` false the purchase is exactly X (as before). With `up_to` true it is as many as the portfolio can pay for, at most X. With no X
        (None) it is as many as the portfolio can pay for: `Problem.max_affordable`, the arithmetic of cash plus everything that may be sold, so the plan has to sell nearly all of it and the tree is
        given a depth of as many steps as there are holdings that can be sold. WHAT TO SELL is always decided by the tree. If not even one share can be paid for, nothing is sold."""
        limits = dict(limits)
        sizes = {k: limits.pop(k) for k in ("max_depth", "branching") if limits.get(k)}
        prices = _prices_of(quotes, list(quotes))      # a quote is the Data Clerk's float: it becomes a Decimal price here, once
        probe = rf.build_problem(portfolio, target, 1, prices, today, only_green, fill, never_liquidate=never_liquidate, also=also)
        cap = probe.max_affordable
        resolved = cap if shares is None else min(shares, cap) if up_to else shares
        info = {"requested": shares, "up_to": up_to, "resolved_shares": resolved, "most_affordable": cap}
        if shares is None:  # everything usable has to go: a plan as deep as the holdings, and a budget to match
            eligible = sum(probe.eligible(h) for h in probe.holdings.values())
            sizes.setdefault("max_depth", max(eligible, 1))
            limits.setdefault("max_calls", 12 + 6 * max(eligible - 3, 0))
        if whole_plans:   # a level is a ROUND of the model, and a round can hold any number of sells: three rounds are enough (propose, then add to the best plans, then once more)
            sizes.setdefault("max_depth", 3)
            auto_depth = False
        if resolved < 1:  # not even one share can be paid for, even by selling everything that may be sold: nothing is sold, and no model is asked
            problem = rf.build_problem(portfolio, target, 1, prices, today, only_green, fill, explore, lean, green_first=green_first, never_liquidate=never_liquidate, whole_plans=whole_plans, also=also)
            return {**rf.answer(problem, [], {"trace": [], "calls": 0, "nodes": 1, "errors": [],
                                              "stop": "selling everything that may be sold cannot pay for even one share: nothing is sold"}, rf.effective_limits(problem)), **info}
        problem = rf.build_problem(portfolio, target, resolved, prices, today, only_green, fill, explore, lean, green_first=green_first, never_liquidate=never_liquidate, whole_plans=whole_plans, also=also, **sizes)
        if "max_depth" not in sizes and auto_depth:  # not asked for: the depth the purchase needs (entry 49), from arithmetic on the holdings: the fewest stocks that could pay for it, plus two
            problem = dataclasses.replace(problem, max_depth=problem.auto_depth())
            limits.setdefault("max_calls", max(rf.MAX_LLM_CALLS, -(-8 * (1 + limits.get("beam_width", rf.BEAM_WIDTH) * (problem.max_depth - 1)) // 5)))   # a deeper tree needs the calls to reach its depth, and room for the re-asks (1.6 times, rounded up: in whole numbers)
            info["auto_depth"] = True
        try:
            return {**rf.search(problem, lambda prob, node, rejected: self._generate(prob, node, rejected), **limits), **info}
        except Exception as error:  # the search catches a failing model call; anything else is a bug we still must not hide behind a traceback in the middle of a desk run
            self.run["error"], self.run["rate_limited"] = f"the search failed ({type(error).__name__})", is_rate_limited(error)
            return {"status": "error", "errors": [self.run["error"]], "target": target, "shares": resolved, **info}

    @traced("account_manager.raise_funds")
    def raise_funds(self, portfolio: Portfolio, target: str, shares: Decimal | None, today: date | None = None, only_green: bool | None = None, fill: str | None = None, explore: bool | None = None,
                    lean: bool | None = None, green_first: bool | None = None, up_to: bool | None = None, never_liquidate: bool | None = None, whole_plans: bool | None = None,
                    also: list[tuple[str, Decimal]] | None = None, **limits) -> dict:
        """Buy `shares` of `target`: find what to sell. Returns the answer of `reasoning.raise_funds.search` plus `quotes` (what the Data Clerk returned), `run` (the model, calls and
        tokens) and `prices_order` (the quote work order, as plain JSON). `status` is `plan`, `funded`, `infeasible`, `no_prices` (the Data Clerk could not quote every symbol: nothing is
        guessed and the model is not called) or `refused` (the request itself is wrong: the reason is in `errors`). `only_green`: only holdings that show a profit may be sold. A stock is
        never sold to pay for more of itself. `fill` (entry 44): a second stock bought with the cash LEFT after the purchase, as many shares as it affords; it is never sold, the sells are lean (no
        step sells more shares than the shortfall needs), and if selling everything cannot pay for `shares` of `target` the status is `infeasible` and nothing is sold. Never raises for a model, API
        or input problem. `shares` None: as many as the portfolio can pay for; `up_to`: as many as it can pay for, at most `shares` (see `_plan`); `green_first`: the score prefers selling the most
        profitable (highest percent gain) holdings first (entry 45). `explore` (entry 44): the generator must give both kinds of step, COVER and PARTIAL, and is asked again (once per node, within the budget) when a kind is missing.
        `also` (entry 142, R10) is [(symbol, whole shares)] of the OTHER purchases of the same run: the plan raises the money for the target and for these in ONE plan, and none of these stocks is ever sold."""
        try:
            target = self._check_purchase(target, shares)
            others = tuple((self._check_purchase(symbol, count), sh.quantity(count)) for symbol, count in (also or []))
            if len({target, *(symbol for symbol, _ in others)}) != 1 + len(others):
                raise ValueError("every purchase of a run is a different stock")
            if fill is not None:
                fill = self._check_purchase(fill, 1)
                if fill == target:
                    raise ValueError("the stock to fill with must be another symbol than the stock to buy")
        except ValueError as error:
            return {"status": "refused", "errors": [str(error)], "quotes": {}, "run": {}}
        only_green = self.only_green if only_green is None else only_green
        never_liquidate = self.never_liquidate if never_liquidate is None else never_liquidate
        explore, lean, green_first, up_to, whole_plans, limits = self._resolve(explore, lean, green_first, up_to, whole_plans, limits)
        self._new_run()
        quotes, prices_order, failed = self._prices([target, *([fill] if fill else []), *(symbol for symbol, _ in others), *portfolio.symbols])
        out = {"quotes": quotes, "prices_order": prices_order, "run": self.run}
        if failed:
            return {**out, "status": "no_prices", "errors": failed}
        result = self._plan(portfolio, target, shares, quotes, today, only_green, fill, explore, lean, green_first, up_to, never_liquidate, whole_plans=whole_plans, also=others, **limits)
        return {**out, **result}

    @traced("account_manager.raise_funds_many")
    def raise_funds_many(self, portfolio: Portfolio, purchases: list[tuple[str, Decimal | None]], today: date | None = None, only_green: bool | None = None, explore: bool | None = None, lean: bool | None = None,
                         green_first: bool | None = None, up_to: bool | None = None, never_liquidate: bool | None = None, whole_plans: bool | None = None, **limits) -> dict:
        """Several purchases, one after the other (entry 43): the Data Clerk is asked ONCE for every price (the targets and the holdings); each purchase is then planned against the portfolio
        the one before left, with the shares it bought marked as bought today (so they cannot be sold in the next plan, the T+1 rule) and its cash carried on. A purchase that cannot be paid for
        (`infeasible`, `error`) changes nothing and the next one is still attempted. Returns {status, purchases: [one result each, with `target` and `shares`], quotes, final: the portfolio at the
        end as plain JSON, run, prices_order}. `status` is `done` (every purchase was attempted), `no_prices` or `refused`. The portfolio given is not changed."""
        try:
            cleaned = [(self._check_purchase(t, n), n) for t, n in purchases]
            if not cleaned:
                raise ValueError("give at least one purchase")
            if len(cleaned) > 1 and any(n is None for _, n in cleaned):
                raise ValueError("'as many as possible' (no number of shares) works for ONE stock: with several, give the number of shares of each (use up_to to buy as many as affordable, at most that number)")
        except ValueError as error:
            return {"status": "refused", "errors": [str(error)], "purchases": [], "quotes": {}, "run": {}}
        only_green = self.only_green if only_green is None else only_green
        never_liquidate = self.never_liquidate if never_liquidate is None else never_liquidate
        explore, lean, green_first, up_to, whole_plans, limits = self._resolve(explore, lean, green_first, up_to, whole_plans, limits)
        self._new_run()
        quotes, prices_order, failed = self._prices([*(t for t, _ in cleaned), *portfolio.symbols])
        out = {"quotes": quotes, "prices_order": prices_order, "run": self.run, "purchases": [], "start": json.loads(portfolio.model_dump_json())}
        if failed:
            return {**out, "status": "no_prices", "errors": failed}
        current = portfolio
        for target, shares in cleaned:
            result = self._plan(current, target, shares, quotes, today, only_green, None, explore, lean, green_first, up_to, never_liquidate, whole_plans=whole_plans, **limits)
            out["purchases"].append({"target": target, "shares": shares, **result})
            if result["status"] in ("plan", "funded"):
                sells = [(x["symbol"], x["shares"]) for x in (result["plan"]["sells"] if result["plan"] else [])]
                current = after_purchase(current, target, result["shares"], result["price"], sells, result["after"]["cash"], today)   # the shares BOUGHT (up to X may be fewer than X), not the shares asked
        return {**out, "status": "done", "final": json.loads(current.model_dump_json())}


def report(result: dict) -> str:
    """What the user reads for a RaiseFunds request: the whole answer, or the reason there is none."""
    if result["status"] in ("refused", "no_prices", "error"):
        lines = [f"RAISE FUNDS  {result['status'].upper().replace('_', ' ')}: " + "; ".join(result.get("errors") or ["no reason given"])]
        if result.get("quotes"):
            lines.append("quotes that did come back: " + ", ".join(f"{s} {q['price']:.2f}" for s, q in result["quotes"].items()))
        return "\n".join(lines)
    run = result["run"]
    text = rf.format_answer(result, quotes=result["quotes"])
    return text + f"\nMODEL    {run['model']}: {run['calls']} call(s), {run['retries']} retr{'y' if run['retries'] == 1 else 'ies'}, {run['tokens_in']} tokens in, {run['tokens_out']} out"


def _shares_table(start: dict, final: dict) -> list[str]:
    """Before and after, per symbol (shares and average cost basis), and the cash: the whole run's effect on the portfolio."""
    before = {p["symbol"]: p for p in start["positions"]}
    after = {p["symbol"]: p for p in final["positions"]}
    rows = [[sym, sh.text(before[sym]["shares"]) if sym in before else "0", sh.text(after[sym]["shares"]) if sym in after else "0",
             f"${after[sym]['cost_basis']:,.2f}" if sym in after else "-"] for sym in [*before, *(s for s in after if s not in before)]]
    rows.append(["cash", f"${start['cash_avail']:,.2f}", f"${final['cash_avail']:,.2f}", ""])
    return rf._table(["symbol", "before", "after", "cost basis after"], rows, right={1, 2, 3})


def report_many(result: dict) -> str:
    """What the user reads for several purchases in a row: each purchase as `report` shows one (the quotes once, at the top), then a summary of the run."""
    if result["status"] != "done":
        return report(result)
    lines, n = [], len(result["purchases"])
    for i, p in enumerate(result["purchases"], 1):
        lines.append(f"\n{'=' * 100}\nPURCHASE {i} of {n}: buy {p['shares']} x {p['target']}\n{'=' * 100}")
        lines.append(rf.format_answer(p, quotes=result["quotes"] if i == 1 else None) if p["status"] in ("plan", "funded", "infeasible")
                     else f"{p['status'].upper()}: " + "; ".join(p.get("errors") or ["no reason given"]))
    rows = []
    for i, p in enumerate(result["purchases"], 1):
        sold = ", ".join(f"{x['symbol']} x{x['shares']}" for x in (p["plan"]["sells"] if p.get("plan") else [])) or "-"
        rows.append([str(i), f"{p['target']} x{p['shares']}", p["status"].upper(), sold, f"${p['cost']:,.2f}" if "cost" in p else "-",
                     f"${p['after']['cash']:,.2f}" if p.get("after") else "-", str(p["plan"]["score"]) if p.get("plan") else "-"])
    lines += [f"\n{'=' * 100}\nSUMMARY OF {n} PURCHASES (in order; each was planned against what the one before left)\n{'=' * 100}",
              *rf._table(["#", "buy", "result", "sold to pay for it", "cost", "cash after", "score"], rows, right={4, 5, 6}),
              "", "PORTFOLIO, start of the run and end of it", *_shares_table(result["start"], result["final"])]
    run = result["run"]
    lines.append(f"\nMODEL    {run['model']}: {run['calls']} call(s), {run['retries']} retr{'y' if run['retries'] == 1 else 'ies'}, {run['tokens_in']} tokens in, {run['tokens_out']} out")
    return "\n".join(lines)


def parse_purchase(text: str, default_shares: int | None) -> tuple[str, Decimal | None]:
    """"MU:2" -> ("MU", Decimal(2)); "MU" -> ("MU", default_shares), which is None, as many as possible, unless --shares was given. The user's typed text is read into a number once, here, at the edge. Raises ValueError, with the reason, for anything else."""
    symbol, _, count = text.partition(":")
    if count and not count.strip().isdigit():
        raise ValueError(f"{text!r}: the number of shares after the colon must be a whole number")
    return symbol.strip(), Decimal(int(count)) if count else None if default_shares is None else Decimal(default_shares)


def wanted_in_dollars(plan) -> bool:
    """Whether the how-much answer of a direct order is money (a total in dollars, a percentage, or an amount for EACH stock) and so gives orders in dollars; an answer in SHARES (`wanted` with no dollars) gives orders in shares."""
    return plan.wanted is None or plan.in_dollars


def _moment(text: str) -> datetime:
    """An aware UTC moment from the text the decisions write ("2026-11-20T21:00:00Z")."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _prices_of(quotes: dict, symbols: list[str]) -> dict[str, Decimal]:
    """{symbol: price} of the symbols that have a usable quote, THE DOOR of a quote: the Data Clerk's `data.quotes` hold the price as the int or float its API client gave, and it becomes a Decimal here, once (`money.price`: above zero, four places,
    half up). A symbol with no quote, one that is not a record, or a price that is not a number above zero (a bool, a string) is left out; the caller says which are missing."""
    prices = {}
    for symbol in symbols:
        quote = quotes.get(symbol)
        if isinstance(quote, dict):
            try:
                prices[symbol] = price_door(quote.get("price"))
            except ValueError:
                pass
    return prices


def holding_lines(portfolio: Portfolio) -> list[str]:
    """One line for each holding of the portfolio (symbol, shares, cost, the day of the latest purchase), or "(no positions)": the listing of `format_account` and of `AccountManager.show_portfolio`."""
    if not portfolio.positions:
        return ["(no positions)"]
    width = max(6, *(len(sh.text(p.shares)) for p in portfolio.positions))                  # a quantity with a fraction is wider than six characters (entry 110)
    return [f"{p.symbol:<6} {sh.text(p.shares):>{width}}  cost {p.cost_basis:,.4f}  acquired {p.acquired or '-'}"
            + (f"  ({sh.text(portfolio.reserved_shares[p.symbol])} set aside for a waiting sale)" if portfolio.reserved_shares.get(p.symbol) else "") for p in portfolio.positions]


def format_account(manager: AccountManager, start: date, end: date) -> str:
    """What the user reads for one saved account: who it is, the cash, the realized gain, the counts, and the holdings."""
    holder = manager.users.get()
    portfolio = manager.list_portfolio()
    gain = manager.calculate_total_gain(start, end)
    lines = [
        f"ACCOUNT {holder.account_number}  {holder.display_name}" + (f"  ({holder.seed_key})" if holder.seed_key else ""),
        f"preferred {holder.preferred_name or '-'}   email {holder.email or '-'}   phone {holder.phone or '-'}",
        f"timezone {holder.timezone}   currency {holder.currency}   status {holder.status}",
        f"risk {holder.risk_tolerance}   mode {holder.trade_mode}   cash {manager.account_balance():,.2f}",
        (f"{start.isoformat()} to {end.isoformat()}: realized gain {gain:,.2f}; trades {manager.number_trades(start, end)}; "
         f"approved {manager.number_approved_recommendations(start, end)}; denied {manager.number_denied_recommendations(start, end)}"),
        "realized gain is the profit booked on sells in that window. Deposits and shares still held are not counted.",
        "PORTFOLIO",
    ]
    lines.extend(holding_lines(portfolio))
    waiting = manager.open_orders()
    if waiting:      # waiting limit orders (entry 140): what they set aside is not available to anything else
        lines += [f"WAITING ORDERS  (cash set aside {portfolio.reserved_cash:,.2f}; available {portfolio.cash_avail:,.2f})", *[f"  {orders.order_line(order)}" for order in waiting]]
    return "\n".join(lines)


def _print_account(args) -> int:
    """The saved account. No model and no quote."""
    import psycopg

    from prometheus.persistence.accounts import open_account_store
    try:
        start = date.fromisoformat(args.gain_from) if args.gain_from else date(2024, 1, 1)
        end = date.fromisoformat(args.gain_to) if args.gain_to else date.today()
        store = open_account_store()
        account_id = int(args.account) if args.account.isdigit() else store.account_id_for_seed(args.account)
        manager = AccountManager(model=object(), model_id="account-book", store=store, account_id=account_id)
        print(format_account(manager, start, end))
    except (ValueError, OSError, RuntimeError, psycopg.Error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RaiseFunds: what to sell from a portfolio to buy one or more stocks, by a Tree of Thought. LIVE when buying: it calls Finnhub for the quotes and OpenRouter for the model. --account only reads the local Postgres account book.")
    parser.add_argument("--buy", action="append", help="the symbol to buy, for example MU, or MU:2 for 2 shares; repeat it for several purchases (done in order)")
    parser.add_argument("--account", help="print one saved account and exit: a sample key (jacob_sample, avery_chen, blake_okonkwo) or an account number. No quotes and no model")
    parser.add_argument("--from", dest="gain_from", help="with --account, the first day of the gain and the counts (default 2024-01-01)")
    parser.add_argument("--to", dest="gain_to", help="with --account, the last day, included (default today)")
    parser.add_argument("--shares", type=int, default=None, help="how many shares of a --buy that has no :N. Left out: AS MANY AS THE PORTFOLIO CAN PAY FOR (one stock only; this can sell nearly everything)")
    parser.add_argument("--exact", action="store_true", help="a number of shares is exact: if the portfolio cannot pay for it, nothing is sold. Without this, N means UP TO N: as many as it can pay for, at most N")
    flag = lambda name, text: parser.add_argument(f"--{name}", action=argparse.BooleanOptionalAction, default=True, help=text + " (on by default; --no-" + name + " turns it off)")
    flag("green-first", "the score prefers selling the most profitable holdings first (highest percent gain) instead of realising the least taxable gain")
    flag("explore", "the beam ranks by score and progress, and a thought of one kind only is asked again")
    flag("lean", "no sell of more shares than the shortfall needs")
    parser.add_argument("--whole-plans", action="store_true", help="a thought is a whole plan (a list of sells): one to three rounds of the model for a purchase. TRIED AND SLOWER (entry 50): off by default")
    parser.add_argument("--portfolio", default=str(DEFAULT_PORTFOLIO), help="a portfolio JSON file (default: the sample portfolio of Jacob Pyrett)")
    parser.add_argument("--only-green", action="store_true", help="(on by default since entry 47) sell only holdings that show a profit: price above cost basis, so no sale takes a loss")
    parser.add_argument("--allow-loss", action="store_true", help="turn off the rule that only holdings in profit are sold: a sale may take a loss")
    parser.add_argument("--allow-liquidate", action="store_true", help="turn off the rule that one share of every holding is kept: a position may be sold out")
    parser.add_argument("--sequential", action="store_true", help="expand the plans of a level one after the other (the default is to call the model for all of them at the same time: same result, less waiting)")
    parser.add_argument("--effort", choices=("low", "medium", "high"), help="the model's reasoning effort (OpenRouter). Default: low for gpt-oss, the model's own for any other. Medium and high only cost time: measured, the same plan")
    parser.add_argument("--max-depth", type=int, help="the most levels: with --no-whole-plans the most stocks sold in a plan (default: chosen from the purchase); with whole plans the most rounds (default 3)")
    parser.add_argument("--branching", type=int, help="the most thoughts asked of the model for each plan (default 3)")
    parser.add_argument("--beam", type=int, help="how many unfinished plans are expanded per level (default 3)")
    parser.add_argument("--max-calls", type=int, help="the most model calls per purchase (default: chosen from the depth)")
    parser.add_argument("--deadline", type=float, help="seconds a purchase may take (default 90; 0 for no limit): after it no new level is started and the best plan COMPLETED so far is the answer (none yet: nothing is sold). A level in flight is waited for")
    parser.add_argument("--fill", metavar="SYMBOL", help="with ONE --buy: also buy as many shares of this stock as the cash LEFT after the purchase affords (the sells are lean: only as much as the purchase needs; "
                        "if selling everything cannot pay for the purchase, nothing is sold)")
    parser.add_argument("--providers", help="the OpenRouter providers to prefer, comma separated, others as fallback (default for gpt-oss: cerebras,groq, about ten times faster than OpenRouter's own choice; 'none': its own choice)")
    parser.add_argument("--model", help="an OpenRouter model id instead of ACCOUNT_MANAGER_MODEL")
    parser.add_argument("--json", action="store_true", help="print the whole result as JSON instead of the report")
    args = parser.parse_args(argv)
    langsmith_tracing.start()   # LangSmith traces the run when LANGSMITH_API_KEY is set (entry 114)
    if args.account:
        if args.buy:
            print("ERROR: --account prints a saved account and does not place a purchase", file=sys.stderr)
            return 1
        return _print_account(args)
    if not args.buy:
        print("ERROR: give --buy SYMBOL, or --account NAME to print a saved account", file=sys.stderr)
        return 1
    try:
        purchases = [parse_purchase(text, args.shares) for text in args.buy]
        portfolio = load_portfolio(args.portfolio)
        agent = AccountManager(model_id=args.model, effort=args.effort or "auto", only_green=not args.allow_loss, never_liquidate=not args.allow_liquidate, green_first=args.green_first,
                               explore=args.explore, lean=args.lean, whole_plans=args.whole_plans, parallel=not args.sequential, up_to=not args.exact,
                               deadline=(None if args.deadline == 0 else args.deadline or 90.0),
                               providers="auto" if not args.providers else None if args.providers == "none" else tuple(args.providers.split(",")))
    except (ValueError, OSError, RuntimeError) as error:  # a bad purchase or portfolio file, or a missing key: the message names the setting, never a value
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    amount = lambda n: str(n) if n else "as many as possible"
    if args.fill and len(purchases) != 1:
        print("ERROR: --fill goes with exactly one --buy", file=sys.stderr)
        return 1
    print(f"RAISE FUNDS for {portfolio.user}: buy {', then '.join(f'{amount(n)} x {s.upper()}' for s, n in purchases)}{f', then as many {args.fill.upper()} as the cash left buys' if args.fill else ''}; holdings {', '.join(f'{p.symbol} x{sh.text(p.shares)}' for p in portfolio.positions)}; "
          f"free cash ${portfolio.cash_avail:,.2f}{'' if args.allow_loss else '; only holdings in profit may be sold (no loss)'}{'' if args.allow_liquidate else '; one share of every holding is kept'}; model {getattr(agent.model, 'model_name', '?')}\n"
          f"(live: one Finnhub quote per symbol, then at most {rf.MAX_LLM_CALLS} model calls per purchase)\n")
    limits = {k: v for k, v in (("max_depth", args.max_depth), ("branching", args.branching), ("beam_width", args.beam), ("max_calls", args.max_calls)) if v}
    started = time.perf_counter()
    if len(purchases) == 1:
        result = agent.raise_funds(portfolio, *purchases[0], fill=args.fill, **limits)
        print(rf.as_json(result) if args.json else report(result))
        print(f"TIME     {time.perf_counter() - started:.0f} seconds ({'sequential' if args.sequential else 'the plans of a level in parallel'}{', reasoning effort ' + args.effort if args.effort else ''}{'' if args.whole_plans else ', one sell per thought'})")
        return 0 if result["status"] in ("plan", "funded") else 1
    result = agent.raise_funds_many(portfolio, purchases, **limits)
    print(rf.as_json(result) if args.json else report_many(result))
    print(f"TIME     {time.perf_counter() - started:.0f} seconds ({'sequential' if args.sequential else 'the plans of a level in parallel'}{', reasoning effort ' + args.effort if args.effort else ''}{'' if args.whole_plans else ', one sell per thought'})")
    return 0 if result["status"] == "done" and all(p["status"] in ("plan", "funded") for p in result["purchases"]) else 1


if __name__ == "__main__":
    sys.exit(main())
