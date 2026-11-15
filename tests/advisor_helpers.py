"""Helpers the Advisor tests share: a metric result object, a metrics dict for one symbol, and the config.

`metric(...)` is a metric result as the Data Clerk writes it. These are inputs for testing a RULE at its threshold (ADV one dollar under the limit, RSI exactly 30): they are not market data,
and the tests of what a recommendation says about SNDK and PG do not use them.
"""
from datetime import date

from prometheus.agents.advisor.config import load_config

TODAY = date(2026, 10, 23)
CONFIG = load_config()


def metric(value, as_of="2026-10-23", note=None, unit="x", source="test"):
    return {"value": value, "unit": unit, "as_of": as_of, "window": "test", "source": source, "inputs_summary": {}, "note": note}


def metrics_of(**values) -> dict:
    """{name: metric result} from keyword values; a value of None is a missing metric with a note (`needs ... bars`)."""
    return {name: metric(v, note=None if v is not None else f"needs bars for {name}") for name, v in values.items()}


def healthy(**override) -> dict:
    """A symbol with every number the gates and quality checks read, all fine; `override` changes some (None makes it missing)."""
    base = dict(last_close=100.0, roe=20.0, operating_margin=15.0, revenue_growth=5.0, debt_to_equity=0.8, current_ratio=1.5, share_count_change=-0.01,
                sbc_to_revenue=0.03, accrual_ratio=0.02, receivables_vs_sales=2.0, inventory_vs_sales=1.0, cfo_negative_with_profit=False,
                risk_factor_change=0.1, days_to_next_earnings=None)
    base.update(override)
    return metrics_of(**base)


def family_input(metrics: dict, events=(), signals=(), regime="trend", quality=None, **kw):
    """A FamilyInput over a metrics dict, with the config of the repo."""
    from prometheus.agents.advisor.families import FamilyInput
    from prometheus.agents.advisor.record import Quality
    return FamilyInput(symbol=kw.pop("symbol", "AAA"), metrics=metrics, events=list(events), signals=list(signals), quality=quality or Quality(multiplier=1.0), regime=regime, config=CONFIG, today=TODAY, **kw)


def fr(family="trend", direction="bullish", edge=0.004, passed=True, **kw):
    """A passed (or not) FamilyResult with an edge, for the ranking tests."""
    from prometheus.agents.advisor.record import FamilyResult
    return FamilyResult(family=family, computable=True, passed=passed, direction=direction if passed else None, edge_20d=edge if passed else None, score=0.8 if passed else None,
                        horizon_bars=kw.pop("horizon_bars", 63), prior_20d=kw.pop("prior_20d", 0.006), **kw)


def scored(symbol="AAA", families=(), metrics=None, votes=(), events=(), signals=(), held=False, sector=None, quality=None, gates=None, regime="trend", rank=True, **kw):
    """A Scored symbol over healthy metrics and passed gates; `rank=True` fills its composite, excess, stance and score."""
    from prometheus.agents.advisor.gates import run_gates
    from prometheus.agents.advisor.ranking import Scored, rank_symbol
    from prometheus.agents.advisor.record import Quality
    metrics = healthy() if metrics is None else metrics
    s = Scored(symbol=symbol, company=None, origin=kw.pop("origin", "subject"), held=held, metrics=metrics, gates=gates or run_gates(metrics, list(events), TODAY, CONFIG.gates), quality=quality or Quality(multiplier=1.0),
               families=list(families), events=list(events), signals=list(signals), votes=list(votes), sector_etf=sector, **kw)
    return rank_symbol(s) if rank else s


# ---------------------------------------------------------------- the whole step, with fakes

import json as _json

from langchain_core.messages import AIMessage

from prometheus.common import work_order as wo
from prometheus.common.portfolio import Portfolio

NEWS = [  # REAL sentences of news/PG.txt, news/F.txt and news/SNDK.txt, copied exactly; the numbering, the dates and the rest of the source dict are the test's
    (1, "PG", "Procter & Gamble (PG) lifted its full-year sales growth outlook to reflect an acquisition, while the company reported Q1 earnings that exceeded analyst expectations.", "2026-10-22T13:00:43Z"),
    (2, "PG", "Procter & Gamble rises after reporting solid organic growth in its product categories.", "2026-10-21T10:00:00Z"),
]


def flat_bars(n=35, close=100.0, volume=1000.0, end="2026-10-23"):
    """[date, close, volume] for the last n weekdays: the `recent_bars` metric of a flat stock."""
    from bar_helpers import weekdays
    from datetime import date as _d
    return [[d.isoformat(), close, volume] for d in weekdays(n, _d.fromisoformat(end))]


def full_metrics(**override) -> dict:
    """Every metric the Advisor asks for, for a stock that passes the gates and the capital_return family; `override` changes some (None makes it missing)."""
    base = dict(last_close=100.0, atr_14=4.0, atr_pct=0.04, ret_1d=0.001, ret_5d=0.01, ret_21d=0.02, volume_ratio_1d=1.1, sma_50=98.0, above_sma_200=True, above_sma_50=True,
                roe=20.0, operating_margin=15.0, revenue_growth=5.0, debt_to_equity=0.8, current_ratio=1.5, share_count_change=-0.02, earnings_yield=0.04,
                payout_coverage=0.6, cash_balance=10e9, vol_63d_ann=0.25, beta_63d=0.7, max_drawdown_63d=-0.05, sbc_to_revenue=0.02,
                accrual_ratio=0.01, receivables_vs_sales=1.0, inventory_vs_sales=1.0, cfo_negative_with_profit=False,
                spy_above_sma_200=True, spy_vol_21d_ann=0.15, spy_drawdown_63d=-0.03, spy_ret_21d=0.02, spy_ret_5d=0.0)
    base.update(override)
    out = metrics_of(**base)
    out["recent_bars"] = metric(flat_bars(), unit="[date, close, volume]")
    return out


def ready_order(subject=("PG",), case=None, sources=NEWS, **kw):
    """A work order whose data is ready: the REAL sentences above as news sources (they are the test's inputs)."""
    from prometheus.agents.data_clerk.data_clerk_agent import Result
    order = wo.new_work_order("Which of these should we buy?", subject=list(subject) if not isinstance(subject, str) else subject, since_days=7, case=case, **kw)
    src = [{"number": n, "chunk_id": f"finnhub:{n}#0", "article_id": f"finnhub:{n}", "chunk": 1, "chunks": 1, "headline": f"Headline {n}", "publisher": "Yahoo", "published_utc": day, "url": f"https://x/{n}",
            "tickers": [t], "text": text, "score": 0.6} for n, t, text, day in sources]
    return wo.attach_result(order, Result(question="q", answer="ok [1]", references=[{"number": 1, "chunk_id": src[0]["chunk_id"]}], sources=src,
                                          tool_calls=[{"name": "retrieve_news", "args": {}, "outcome": "ok"}]))


class FakeClerk:
    """The Data Clerk as the Advisor sees it: `request_metrics` and `tools.discover_tickers`. `metrics` is {symbol: metrics dict}; a symbol that is not in it gets no result."""

    def __init__(self, metrics: dict, discover=None, fail=False):
        self.metrics, self.discover, self.fail, self.asked = metrics, discover or [], fail, []
        self.tools = self

    def discover_tickers(self, phrase, from_date=7, top=10):
        self.asked.append(("discover", phrase))
        return {"tickers": self.discover}

    def request_metrics(self, order, today=None):
        self.asked.append(("metrics", list(order.subject), [m["name"] for m in order.metrics], order.sector))
        if self.fail:
            order.status, order.errors = "data_failed", [*order.errors, "boom"]
            return order
        order.data = wo.DataResult(metrics={s: self.metrics[s] for s in order.subject if s in self.metrics})
        order.status = "data_ready"
        return order


class Scripted:
    """Anything with invoke: returns the next scripted reply (a string, an AIMessage or an Exception to raise) and records what it was asked."""
    model_name = "scripted/advisor"

    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def invoke(self, messages):
        self.seen.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, AIMessage) else AIMessage(content=reply, usage_metadata={"input_tokens": 500, "output_tokens": 100, "total_tokens": 600})


def plan_reply(*companies) -> str:
    return _json.dumps({"companies": list(companies)})


def company(symbol="PG", name="Procter & Gamble", signals=(), events=(), drop_cause=None, model_score=70, archetypes=("capital_return",)):
    return {"symbol": symbol, "company": name, "archetypes": list(archetypes), "signals": list(signals), "events": list(events), "drop_cause": drop_cause, "model_score": model_score}


PG_QUOTE = "Procter & Gamble (PG) lifted its full-year sales growth outlook to reflect an acquisition"
PG_SIGNAL = {"source": 1, "direction": "bullish", "quote": PG_QUOTE}
PG_EVENT = {"class": "guidance_raise", "source": 1, "quote": PG_QUOTE}


class MemoryLedger:
    """A ledger in memory with the Ledger's methods the Advisor uses."""

    def __init__(self, records=()):
        self.records, self.problems = list(records), []

    def append(self, record):
        self.records.append(record)

    def recent(self, symbol, family=None, since=None):
        return [r for r in self.records if r.get("symbol") == symbol and (family is None or (r.get("trade") or {}).get("family") == family)]


def thesis_reply(symbol="PG", thesis="Open Procter & Gamble on its capital return: the expected excess return is an unmeasured prior.", bull=(("It lifted its outlook.", "[1]"),), bear=()) -> str:
    return _json.dumps({"theses": [{"symbol": symbol, "thesis": thesis, "bull_case": [{"text": t, "ref": r} for t, r in bull], "bear_case": [{"text": t, "ref": r} for t, r in bear]}]})


def advisor_with(*replies, metrics=None, ledger=None, discover=None, clerk=None):
    from prometheus.agents.advisor.advisor import Advisor
    model = Scripted(*replies)
    advisor = Advisor(model=model, clerk=clerk or FakeClerk(metrics if metrics is not None else {"PG": full_metrics()}, discover), ledger=ledger if ledger is not None else MemoryLedger())
    return advisor, model


EMPTY_PORTFOLIO = Portfolio(user="t", cash_avail=1000.0)
