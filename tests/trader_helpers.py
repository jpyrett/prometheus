"""Helpers the Trader tests share (Trader v1 spec section 18): a decided work order, a Data Clerk that answers `request_quotes` and `request_bars`, an Account Manager over it, and a scripted model.

    order = decided()                                           # the captured PG_open order, reviewed by the Critic and decided (yes): 44 shares of PG (the Account Manager's size for an account of $74,481.40), reference 167.10, zone 165.4559 to 168.7441
    order = synthetic(row(symbol="PG", shares=10, price=100.0), row(symbol="MSFT", action="reduce", shares=2, price=400.0))   # a made-up decided order, one row per decision, for the cases the capture cannot give
    m = manager(quotes={"PG": 167.10}, bars={"PG": bars(...)})  # a real AccountManager(model=object(), clerk=MarketClerk(...), profile=...): only its quote and bar requests and its portfolio methods are used
    trader = Trader(model=ScriptedModel([call("verify_request"), say("{...}")]), ledger=MemoryLedger())

Everything is a fake: no network, no model, no paid API. The numbers of `synthetic` and `bars` are INPUTS for testing a rule at its boundary (a price exactly on a limit), not market data.
"""
import json
from datetime import date, datetime, timedelta
from itertools import count

from advisor_helpers import MemoryLedger  # noqa: F401  (re-exported: the Trader's ledger in tests)
from critic_helpers import NOW, at, captured, make_critic, portfolio, profile, view  # noqa: F401
from langchain_core.messages import AIMessage

from prometheus.agents.account_manager import decisions
from prometheus.agents.account_manager.account_manager import AccountManager
from prometheus.agents.critic.config import load_config as load_critic_config
from prometheus.agents.trader import tickets as tk
from prometheus.agents.trader.broker import PaperBroker, Prices
from prometheus.agents.trader.config import TraderConfig, load_config
from prometheus.agents.trader.simulator import entry_date
from prometheus.common.market_calendar import eastern_date
from prometheus.common import work_order as wo
from prometheus.common.portfolio import Portfolio

USER = "Jacob Pyrett"
VALID_UNTIL = "2026-11-05T16:30:00Z"        # the approval of the captured PG_open review: 11:30 Eastern, an hour after NOW
_ids = count(1)


# ---------------------------------------------------------------- the settings

def config(**sections) -> TraderConfig:
    """The shipped settings with some changed: `config(orders={"allow_market": False}, limits={"max_trades": 1}, mode="backtest")`. A dict updates one section, anything else replaces a top-level value."""
    base = load_config()
    return base.model_copy(update={name: getattr(base, name).model_copy(update=values) if isinstance(values, dict) else values for name, values in sections.items()})


# ---------------------------------------------------------------- the model

def call(name: str, **args) -> AIMessage:
    """A model turn that asks for one tool."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call{next(_ids)}"}], usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110})


def calls(*asks) -> AIMessage:
    """A model turn that asks for SEVERAL tools at once: each ask is (name, args)."""
    return AIMessage(content="", tool_calls=[{"name": n, "args": a, "id": f"call{next(_ids)}"} for n, a in asks], usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110})


def say(text: str) -> AIMessage:
    """A model turn that answers."""
    return AIMessage(content=text, usage_metadata={"input_tokens": 120, "output_tokens": 20, "total_tokens": 140})


def answer(summary: str, *trades) -> str:
    """The JSON the Trader's model must end with: {"summary": ..., "trades": [{"trade_id", "status"}, ...]}; each trade is (trade_id, status)."""
    return json.dumps({"summary": summary, "trades": [{"trade_id": i, "status": s} for i, s in trades]})


class ScriptedModel:
    """Anything with bind_tools and invoke. `seen` records every request: ("tools" or "plain", the messages sent). A script item that is an Exception is raised."""
    model_name = "scripted/trader"

    def __init__(self, script=()):
        self.script, self.seen, self.bound = list(script), [], []
        self.ran_dry = False                                   # the model was asked for a turn the script did not have: the Trader swallows a model's failure, so a test must look at this

    def bind_tools(self, tools):
        outer = self
        outer.bound.append([t.name for t in tools])

        class Bound:
            def invoke(self, messages):
                outer.seen.append(("tools", list(messages)))
                return outer._next()

        return Bound()

    def invoke(self, messages):
        self.seen.append(("plain", list(messages)))
        return self._next()

    def _next(self):
        if not self.script:
            self.ran_dry = True
            raise AssertionError("the scripted model was asked for more turns than the test gave it")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def texts(self) -> str:
        """Everything the model was ever shown, as one text (the system prompt, the user message and every tool result)."""
        return "\n".join(str(m.content) for _, messages in self.seen for m in messages)


class BrokenModel(ScriptedModel):
    """A model whose every call raises."""

    def __init__(self, error=None):
        super().__init__()
        self.error = error or RuntimeError("the provider is down")

    def _next(self):
        raise self.error


# ---------------------------------------------------------------- the Data Clerk

def bar(day, o, h, l, c, v=1000):
    return {"date": day, "open": o, "high": h, "low": l, "close": c, "volume": v}


def weekday_bars(start: str, rows, ) -> list[dict]:
    """Bars on consecutive weekdays from `start`: each row is (open, high, low, close)."""
    day, out = date.fromisoformat(start), []
    for r in rows:
        while day.weekday() >= 5:
            day += timedelta(days=1)
        out.append(bar(day.isoformat(), *r))
        day += timedelta(days=1)
    return out


class MarketClerk:
    """The Data Clerk as the Account Manager asks it for prices (`request_quotes`, `request_bars`). It records every order it was asked.

    quotes   {symbol: price or a quote dict}: a symbol that is not there is a failed quote (the order is `data_failed`, the quotes that did come back stay on it, as the real clerk does)
    bars     {symbol: [bar, ...]}, oldest first: the order's range is applied, as the real store does
    fail     names of requests that raise ("quotes", "bars")
    refuse   names of requests the clerk answers with `data_failed` and no data ("quotes", "bars")
    """

    def __init__(self, quotes=None, bars=None, fail=(), refuse=()):
        self.quotes, self.bars, self.fail, self.refuse = dict(quotes or {}), dict(bars or {}), set(fail), set(refuse)
        self.orders: list = []

    def request_quotes(self, order):
        self.orders.append(order)
        if "quotes" in self.fail:
            raise RuntimeError("the quotes are down")
        if "quotes" in self.refuse:
            order.status, order.errors = "data_failed", [*order.errors, "the quote request was refused"]
            return order
        found, problems = {}, []
        for symbol in order.subject:
            if symbol in self.quotes:
                q = self.quotes[symbol]
                found[symbol] = q if isinstance(q, dict) else {"symbol": symbol, "price": q, "as_of_utc": "2026-11-05T15:29:00Z", "source": "fake"}
            else:
                problems.append(f"{symbol}: no quote (RuntimeError: unknown symbol)")
        order.data = wo.DataResult(quotes=found, problems=problems)
        order.errors = [*order.errors, *problems]
        order.status = "data_failed" if problems else "data_ready"
        return order

    def request_bars(self, order):
        self.orders.append(order)
        if "bars" in self.fail:
            raise RuntimeError("the bars are down")
        if "bars" in self.refuse:
            order.status, order.errors = "data_failed", [*order.errors, "the bars request was refused"]
            return order
        start, end = order.bars["start"], order.bars["end"]
        found = {s: [b for b in self.bars.get(s, []) if start <= b["date"] <= end] for s in order.subject}
        order.data = wo.DataResult(bars=found, notes=[f"{s}: no stored bar from {start} to {end}" for s, v in found.items() if not v])
        order.status = "data_ready"
        return order

    def asked(self, kind: str) -> list[list[str]]:
        """The symbols of each order of one kind ("quotes" or "bars") that was asked for."""
        action = {"quotes": "fetch_quote", "bars": "fetch_bars"}[kind]
        return [list(o.subject) for o in self.orders if o.action_type == action]


def manager(quotes=None, bars=None, clerk=None, user=USER, **options) -> AccountManager:
    """A real Account Manager over a MarketClerk: its profile is Jacob's (Ask mode), it never calls a model, and its quote and bar requests go to the fake clerk."""
    return AccountManager(model=object(), model_id="account-book", clerk=clerk or MarketClerk(quotes, bars), profile=options.pop("profile", None) or profile(user=user), **options)


def account(cash=1000.0, user=USER, **positions) -> Portfolio:
    """A portfolio of Jacob's: `cash` and one position per keyword (`MSFT=(5, 400.0)` is 5 shares at a cost of 400.00; a bare number is that many shares at 100.00), all bought long ago, so all sellable."""
    rows = []
    for symbol, spec in positions.items():
        shares, cost = spec if isinstance(spec, tuple) else (spec, 100.0)
        rows.append({"symbol": symbol, "shares": shares, "cost_basis": cost, "acquired": "2025-01-02"})
    return Portfolio(user=user, cash_avail=cash, positions=rows)


# ---------------------------------------------------------------- work orders

def decided(name: str = "PG_open", answer: bool | None = True, mode: str | None = None, now: datetime = NOW) -> wo.WorkOrder:
    """The captured order of `critic_helpers.captured`, reviewed by `make_critic` and decided by `decisions.decide_trades`. With `PG_open` and a yes that is one permitted purchase: 44 shares of PG, reference price
    167.10, entry zone 165.4559 to 168.7441, plan limit 167.10, stop 157.2351, target 168.3783, time stop 2026-12-31, approval valid until 2026-11-05T16:30:00Z, `NOW` 10:30 Eastern that day. `answer` is what the
    user says to the Ask prompt: True (yes_user), False (no_user_denied), None (pending_user).

    The purchase is the Account Manager's size (design choices entry 80), for an account of $74,481.40 in cash (what the sample portfolio is worth at the capture's prices): the profile's 10% position limit is 44 shares
    at 167.10. The Trader tests then give the Trader their OWN account (`account(51.23)`, the sample's cash, to prove the funding and the `insufficient_cash` rejection): the Trader checks the money again when it
    places each order, whatever the decision was sized for."""
    order = captured(name)
    make_critic(order).review_recommendations(order, view(), now)
    return decisions.decide_trades(order, Portfolio(user=USER, cash_avail=74481.40), profile(), load_critic_config().verdict.auto_min_score, lambda card: answer, now, mode)


def row(symbol="PG", action="open", shares=10, price=100.0, zone=None, limit=None, will_trade="yes_auto", approved=True, valid_until=VALID_UNTIL, stop=90.0, target=110.0, time_stop="2026-12-31", **extra) -> dict:
    """One decision of a `synthetic` order. `zone` None is (price - 1, price + 1) for a purchase and no zone for a sale; `limit` None is a plan without a limit price (a market order)."""
    return {"symbol": symbol, "action": action, "shares": shares, "price": price, "zone": zone, "limit": limit, "will_trade": will_trade, "approved": approved, "valid_until": valid_until, "stop": stop, "target": target,
            "time_stop": time_stop, **extra}


def synthetic(*rows, status="decided") -> wo.WorkOrder:
    """A decided work order made of nothing but the fields the Trader reads, one decision per row (record ids r1, r2 ...): the recommendation (identity, the trade plan), the Critic's review (approval, deal price) and the
    Account Manager's decision. Used where the captured order has only the one purchase."""
    order = wo.new_work_order("trade the recommendations", subject=[r["symbol"] for r in rows])
    order.status = status
    order.schema_version = 3
    for n, r in enumerate(rows, 1):
        rid, buy = r.get("record_id", f"r{n}"), r["action"] != "reduce"
        zone = r["zone"] if r["zone"] is not None else [r["price"] - 1, r["price"] + 1] if buy else None
        order.recommendations.append({"symbol": r["symbol"], "company": f"{r['symbol']} Holdings", "identity": {"record_id": rid}, "action": r["action"],
                                      "trade": {"entry_zone": zone, "entry_mid": r["price"] if buy else None, "limit_price": r["limit"], "order_style": "limit" if r["limit"] else "market", "stop_price": r["stop"],
                                                "target_price": r["target"], "time_stop_date": r["time_stop"]}})
        order.reviews.append({"record_id": rid, "review_id": f"rv{n}", "symbol": r["symbol"], "action": r["action"], "approved": r["approved"], "valid_until_utc": r["valid_until"],
                              "timing": {"deal_price": r["price"], "price_basis": "quote"}})
        order.trade_decisions.append({"record_id": rid, "review_id": f"rv{n}", "symbol": r["symbol"], "action": r["action"], "shares": r["shares"], "est_cost": 0.0, "mode": "auto", "will_trade": r["will_trade"],
                                      "funds_avail": True, "funds_note": None, "asked_user": False, "escalated": False, "reason": "", "decided_utc": "2026-11-05T15:30:00Z"})
    return order


def plan(target="PG", *sells, status="plan") -> dict:
    """A RaiseFunds answer, as `AccountManager.raise_funds` gives it, with the fields the Trader reads: each sell is (symbol, shares, price)."""
    proceeds = lambda n, p: round(n * p, 2) if isinstance(n, (int, float)) and isinstance(p, (int, float)) else None     # a test may give a sell a bad count or price on purpose
    lines = [{"symbol": s, "shares": n, "price": p, "proceeds": proceeds(n, p), "basis": 100.0, "closes_position": False} for s, n, p in sells]
    return {"status": status, "target": target, "shares": 0, "plan": {"sells": lines, "proceeds": sum(x["proceeds"] or 0 for x in lines)} if sells else None, "errors": []}


def say_yes(card):
    return True


def dumps(order: wo.WorkOrder) -> str:
    """The order as JSON, for 'it did not change' tests."""
    return order.model_dump_json()


def other_agents_work(order: wo.WorkOrder) -> str:
    """The bytes of what the Advisor, the Critic and the Account Manager wrote (rule T10): the recommendations, the reviews and the decisions."""
    return json.dumps({"recommendations": order.recommendations, "reviews": order.reviews, "trade_decisions": order.trade_decisions}, sort_keys=True, default=str)



# ---------------------------------------------------------------- a broker, for the tests of the broker and the tools

def paper_prices(quotes: dict, now: datetime = NOW) -> Prices:
    return Prices("paper", eastern_date(now), quotes=dict(quotes))


def backtest_prices(bars: dict, now: datetime = NOW) -> Prices:
    """The snapshot of a backtest over `bars`: the entry date found as the Trader finds it (`simulator.entry_date`)."""
    entry = entry_date(now, bars)
    return Prices("backtest", entry or eastern_date(now), bars=bars, entry_date=entry)


def make_broker(order, portfolio=None, quotes=None, bars=None, now: datetime = NOW, cfg: TraderConfig | None = None, funding=None, mode: str = "paper", consent: bool = True, cash_hint: str = "", loaded=None, acct=None,
                book: bool = False):
    """(broker, trades) for a decided order: the tickets the Trader builds (`tickets.build_tickets`), the funding sales of `funding` (a RaiseFunds answer) when `consent`, and a broker over one snapshot of prices
    (`quotes` for paper, `bars` for backtest). `permit` is `verify_trade` over the order, as the Trader wires it. The broker's `account` is a REAL Account Manager (`acct`, or `manager(...)`) whose trading run is
    open on `portfolio` (`trading_portfolio(portfolio, book=book)`): the funds are ITS, since the per-trade handshake (design choices entry 81), so a test reads them with `acct_now(broker)`."""
    cfg, portfolio = cfg or config(), portfolio or account()
    found = tk.build_tickets(order, portfolio, now, cfg, mode) if loaded is None else list(loaded)     # `loaded`: the trades of an earlier run, as the Trader loads them from the order
    if funding is not None and consent and loaded is None:
        target, legs, _ = tk.funding_legs(funding, found, portfolio, eastern_date(now))
        if target is not None:
            found = [*found, *tk.funding_trades(target, legs, cfg, mode)]
    prices = paper_prices(quotes or {}, now) if mode == "paper" else backtest_prices(bars or {}, now)
    acct = acct or manager(user=portfolio.user)
    acct.trading_portfolio(portfolio, book=book)
    broker = PaperBroker(cfg, mode, acct, order, prices, found, lambda t: tk.permission(order, t.record_id, now), now, cash_hint=cash_hint)
    return broker, found


def acct_now(broker) -> Portfolio:
    """The account as the Account Manager reports it NOW: what a test reads where the broker used to have a portfolio of its own (it has none any more)."""
    return broker.account.current_portfolio()


class FakeNotifier:
    """The Trader's one-shot notifier as the tools see it: reports once, gives the first answer again, closes the broker."""

    def __init__(self, broker=None):
        self.broker, self.done, self.calls, self.answer = broker, False, 0, None

    def __call__(self):
        self.calls += 1
        if not self.done:
            self.done = True
            if self.broker is not None:
                self.broker.closed = True
            self.answer = {"cash_before": 0.0, "cash_after": 0.0, "filled": 0, "rejected": 0, "active": 0, "booked": False}
        return self.answer
