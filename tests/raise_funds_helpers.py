"""REAL data for the tests of raising funds for a purchase the cash does not cover (design choices entry 142; design/direct_raise_funds_spec.md). No invented stock data and no stand-in agent.

    ACCOUNT        the REAL account 6553 (Jacob's, read only) as it was on 2026-11-21 with the REAL Finnhub quotes of ANT and its twelve holdings and the REAL answers of Finnhub's symbol search for ANT and GOOGL
                   (`tests/fixtures/raise_funds/account_6553.json`, made by `scripts/capture_raise_funds_account.py`)
    FORMS          the REAL replies of the Account Manager's model to the requests of the spec (`tests/fixtures/session_real/raise_funds_forms.json`, `scripts/capture_raise_funds_forms.py`)
    plans          the REAL replies of the Account Manager's model (the thought generator of RaiseFunds) to the plans these tests need, recorded by the CONTENT of the messages it was asked
                   (`tests/fixtures/raise_funds/plans.json`, `scripts/capture_raise_funds_plans.py`): a test gives the reply back for the same question, and fails for a question nobody recorded

A test makes a THROWAWAY Postgres account the same as 6553 (`make_account`: the cash, and every position bought on its day at its average cost, through the book's own transaction) and deletes it at the end; the three sample
accounts and 6553 itself are never written. The Data Clerk is the real one: its quotes are the recorded Finnhub answers given back through its own `get_quote` (the tool the Finnhub call lives in), its symbol directory the recorded one.
The Trader is the real Trader with no model (the sweep places the orders), the Account Manager the real one, the graph the real LangGraph desk.
"""
import hashlib
import json
import threading
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from prometheus.agents.account_manager.account_manager import AccountManager
from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
from prometheus.agents.trader.trader import NullLedger, Trader
from prometheus.common import shares as sh
from prometheus.common.money import money as money_door
from prometheus.common.money import price as price_door
from prometheus.common.portfolio import trade_amount
from prometheus.langgraph import desk
from prometheus.prompts import account_manager_request as request_prompt

FIXTURES = Path(__file__).parent / "fixtures"
ACCOUNT = json.loads((FIXTURES / "raise_funds" / "account_6553.json").read_text(encoding="utf-8"))
QUOTES = ACCOUNT["quotes"]
FORMS = json.loads((FIXTURES / "session_real" / "raise_funds_forms.json").read_text(encoding="utf-8"))["requests"]
PLANS_FILE = FIXTURES / "raise_funds" / "plans.json"
NOW = datetime(2026, 11, 20, 21, 10, tzinfo=timezone.utc)      # a few minutes after the quotes were recorded (their `as_of_utc` is 2026-11-20T21:00:00Z): every position was bought on 2026-11-13 or 2026-11-18, so all of it is sellable (T+1)
ASK = "Buy 2 shares of ANT and raise funds if needed"
WAITING = "Buy 2 shares of ANT at $85 or less and raise funds if needed"      # entry 143: ANT is $88.91, above the ceiling, so it WAITS; the recorded day's low is $83.05
WAITING_PLAIN = "Buy 2 shares of ANT at $85 or less"
FALLEN = {"ANT": {**QUOTES["ANT"], "price": QUOTES["ANT"]["low"]}}      # ANT later in the session: the recorded quote with its own recorded day low as the price (real numbers: nothing is invented)
PLAIN = "Buy 2 shares of ANT"
CASH = money_door(Decimal(str(ACCOUNT["cash"])))
ANT = price_door(Decimal(str(QUOTES["ANT"]["price"])))
COST = trade_amount(2, ANT)
SHORT = COST - CASH
BUDGET = Decimal("2000.00")      # a budget under what the typed shares of the recommendation path cost (R10): the pot is the budget


def plans() -> dict:
    """The recorded replies of the thought generator by the content of the question (empty before the first capture)."""
    return json.loads(PLANS_FILE.read_text(encoding="utf-8"))["replies"] if PLANS_FILE.exists() else {}


def key_of(messages) -> str:
    """What identifies a question to the thought generator: every message of the conversation, in order, with its role."""
    text = "\n--\n".join(f"{type(m).__name__}: {m.content}" for m in messages)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Model:
    """The Account Manager's model: its REAL recorded replies given back. Reading the request: the reply to the typed words (`FORMS`), in order. RaiseFunds' thought generator: the reply recorded for the same question (`plans`);
    a question nobody recorded fails the test (a changed prompt or a changed plan needs a new capture: `scripts/capture_raise_funds_plans.py`, which gives a `live` model: a question nobody recorded is asked of it and its reply is kept)."""

    model_name = "openai/gpt-oss-120b"

    def __init__(self, typed: str | None = None, recorded: dict | None = None, live=None):
        self.typed, self.recorded, self.live = typed, plans() if recorded is None else recorded, live
        self.reading = list(FORMS[typed]["replies"]) if typed in FORMS else []
        self.asked: list[str] = []         # the keys of the questions RaiseFunds asked
        self.calls = 0
        self.lock = threading.Lock()

    def invoke(self, messages):
        with self.lock:
            self.calls += 1
            if messages[0].content == request_prompt.SYSTEM:
                if not self.reading:
                    raise AssertionError(f"the model was asked to read {self.typed!r} more times than it has recorded replies")
                return SimpleNamespace(content=self.reading.pop(0), usage_metadata={"input_tokens": 10, "output_tokens": 10})
            key = key_of(messages)
            self.asked.append(key)
            if key not in self.recorded:
                if self.live is None:
                    raise AssertionError("RaiseFunds asked the model a question nobody recorded: capture it with scripts/capture_raise_funds_plans.py")
                reply = self.live.invoke(messages)
                self.recorded[key] = reply.content if isinstance(reply.content, str) else ""
            return SimpleNamespace(content=self.recorded[key], usage_metadata={"input_tokens": 10, "output_tokens": 10})


def clerk(quotes: dict | None = None) -> DataClerk:
    """The real Data Clerk: its symbol directory is Finnhub's recorded answers (a query nobody recorded fails the test), and its `get_quote` gives back the recorded Finnhub quote of the symbol. `quotes` ({symbol: a quote of the recorded shape})
    replaces the quote of those symbols: a test that needs the price to FALL later in the session gives the recorded quote with its `price` set to that quote's own recorded day `low`."""
    directory = {**ACCOUNT["directory"], **json.loads((FIXTURES / "finnhub_company_search.json").read_text(encoding="utf-8"))["queries"]}

    def search(query: str):
        if query not in directory:
            raise AssertionError(f"the Data Clerk asked the symbol directory for {query!r}, which was not recorded")
        return directory[query]

    real = DataClerk(model=object(), symbol_search=search)
    real.lookup_interval = 0

    source = {**QUOTES, **(quotes or {})}

    def quote(symbol: str) -> dict:
        if symbol not in source:
            raise RuntimeError(f"no recorded quote for {symbol}")
        return dict(source[symbol])

    real.tools.get_quote = quote
    return real


def make_account(store, made: list, name: str = "Raise Funds Copy", mode: str = "ask", drop: tuple[str, ...] = (), only: tuple[str, ...] | None = None, cash: Decimal | None = None) -> int:
    """A THROWAWAY account the same as 6553 (the cash, and each position bought on its day at its average cost: the book's own transaction), appended to `made` so that the test deletes it. `drop` leaves positions out, `only` keeps just those,
    `cash` (a Decimal, the account's own cash and nothing about the market) makes the account hold that much cash instead of 6553's."""
    positions = [p for p in ACCOUNT["positions"] if p["symbol"] not in drop and (only is None or p["symbol"] in only)]
    spent = sum((trade_amount(sh.quantity(Decimal(str(p["shares"]))), price_door(Decimal(str(p["cost_basis"])))) for p in positions), Decimal("0.00"))
    account_id = store.create_account(name, "moderate", trade_mode=mode, age=40)
    made.append(account_id)
    store.add_funds(account_id, spent + (CASH if cash is None else money_door(cash)), date(2026, 11, 10))
    for p in positions:
        store.book_trade(account_id, p["symbol"], "buy", sh.quantity(Decimal(str(p["shares"]))), price_door(Decimal(str(p["cost_basis"]))), date.fromisoformat(p["acquired"]))
    return account_id


def manager(store, account_id: int, typed: str | None = None, model=None, **options) -> AccountManager:
    """The real Account Manager over the real Data Clerk and this account; its model gives back the real recorded replies (`Model`)."""
    return AccountManager(model=model or Model(typed), model_id="recorded", clerk=clerk(), store=store, account_id=account_id, **options)


def trader(**options) -> Trader:
    return Trader(use_model=False, ledger=NullLedger(), **options)


def never(name):
    """A factory that FAILS THE TEST if the desk builds the agent: a direct order must not reach it."""
    def factory(order):
        raise AssertionError(f"a direct order must not reach the {name}")
    return factory


def run_desk(store, account_id: int, typed: str, approver=None, mode: str | None = None, model=None, sink: list | None = None, **options):
    """The REAL LangGraph desk over the real Account Manager (the model's replies are the recorded real ones), the real Data Clerk (recorded quotes), the real Trader with no model and the real Postgres book. `sink` collects the
    Account Managers the desk built (to read what their model was asked)."""
    request = {"ask": {"text": typed, "default_count": 3, "fetch": False}, "account": {"account": account_id, **({"mode": mode} if mode else {})}, "trade": {"mode": "paper", "book": True, "model": False}}

    def funds_factory():
        built = manager(store, account_id, typed, model=model)
        if sink is not None:
            sink.append(built)
        return built

    return desk.run_desk(request, clerk_factory=never("Data Clerk's node"), advisor_factory=never("Advisor"), critic_factory=never("Critic"), funds_factory=funds_factory, with_critic=True, with_trader=True,
                         trader_factory=lambda order: trader(), store_factory=lambda: store, approver=approver, clock=lambda: NOW, **options)


class Cards:
    """An approver that records every card it is given and answers by the card's kind: `answers` is {"raise_funds": True, "funding": True, ...} (a kind not in it is not expected: it answers None, no answer, and is listed)."""

    def __init__(self, **answers):
        self.answers, self.cards = answers, []

    def __call__(self, card: dict):
        self.cards.append(card)
        return self.answers.get(card.get("kind") or "approve")

    @property
    def kinds(self) -> list[str]:
        return [c.get("kind") or "approve" for c in self.cards]


# ---------------------------------------------------------------- the recommendation path (entry 142, R10 and R11): the real review of a real session, the real sample portfolio, the stored closes of Friday 2026-11-13

def reviewed_order():
    """The REAL work order of `tests/fixtures/session_real/bob_jones_oil.json` (a real session of 2026-11-14) as the Critic left it: the real recommendations and reviews (XOM and EOG approved, APA denied), not decided yet."""
    import copy

    import handshake_helpers as hh
    from prometheus.common import work_order as wo

    order = wo.WorkOrder.model_validate(copy.deepcopy(json.loads(hh.CAPTURE.read_text(encoding="utf-8"))["order"]))
    order.status, order.trade_decisions, order.trades, order.trader_runs = "reviewed", [], [], []
    return order


def sample_manager(model=None, **options) -> AccountManager:
    """The real Account Manager over the real Data Clerk (the stored closes of the pinned Friday) with the real session's profile; its model gives back the real recorded replies."""
    import handshake_helpers as hh

    return AccountManager(model=model or Model(), model_id="recorded", clerk=hh.clerk(), profile=hh.PROFILE, **options)
