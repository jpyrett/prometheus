"""The Time Machine: the REAL desk replayed at past decision days (design/proving_ground_spec.md section 7; P1, P2, P4, P5, P6, P7, P14; design choices entry 126).

    request = TimeMachineRequest(prompt="Buy the 3 most recommended oil stocks with $3,000, split evenly", cash="10000", start=date(2026, 10, 19), end=date(2026, 11, 13), cadence="weekly")
    run = run_time_machine(request, emit=print_event, should_stop=lambda: False)        # a `ProvingRun`, saved in the store
    python -m prometheus.proving time-machine --prompt "..." --cash 10000 --from 2026-10-19 --to 2026-11-13 --cadence weekly

TWO PHASES, and the page shows both.
PHASE 1, RESEARCH, in parallel (7.6). Every decision day that is not in the research cache is researched at once by a pool of `parallel_days` threads. A task runs in its OWN copy of the context (`contextvars.copy_context()`), binds
`trading_clock.as_of(day)` inside it, builds its own Data Clerk (`allow_fetch=False`: P6), Advisor and Critic over EMPTY throwaway ledgers (so one day's research depends on no other day's, 7.3) and the REAL `build_desk` graph, and
lets it run the desk as far as the Critic: `graph.stream` is stopped when the Critic's node has finished, before the Account Manager's decision (the walk makes the decision with the account's own money). What comes out is the work order as
the Critic left it, which is CACHED (7.4) under a key that holds everything that makes the research what it is: the request as the Account Manager read it, the day, the risk, the two switches, the config versions, the models and the
stored data's stamp. A rerun with another amount of cash, another range or another cadence reuses every day it shares. A day whose research failed is `failed` and the run goes on; a provider's rate limit (429) backs the pool off.
SPEED (P14, `speed.py`): a run is FAST (the default of `time_machine.mode`) or FAITHFUL (`TimeMachineRequest.speed`, `--faithful`). Nothing a program computes differs: the speed only decides how the models are asked (a lower reasoning effort, capped replies, shorter waits and the
fastest provider of the same model in the fast mode; the live desk's own settings in the faithful one), so a fast research is cached apart from a faithful one (`research_key`) and every fast run carries the caveat `fast_models`. All the days are researched at once
(`parallel_days`, up to the config's cap) and the records of one Critic review are checked at once (`Critic(parallel=...)`); both give the sequential result. Every day and every model call is TIMED (`ProvingRun.timing`: the seconds of each desk node, each call's model, seconds and
tokens), and the feedback to LangSmith is posted in the background AFTER the result is emitted and saved (`tracing.post_in_background`; a command waits for it with `tracing.wait_for_posts`).
What is fetched ONCE per run (`memo.Memo`): the request's reading (one model call) and its security screening, Finnhub's fundamentals of each symbol and the symbol directory's lookups (today's, whatever the day: P3).
PHASE 2, THE WALK, in date order, no model: for each trading day from the start to the last usable bar the account's fills of that day come in, `SimPortfolio.step` applies the plan's exits (P5), and on a decision day the Account Manager
decides and sizes in Auto mode with the account's own money (`decide_trades`: the pot, the dollars, the handshake: P7 unchanged) and the Trader fills on the BacktestVenue at the NEXT open (P4, `use_model=False`); then the day is marked.
The Advisor's COOLDOWN (not the same family on the same company while the first plan stands) is the Advisor's own function (`ranking.cooldown_of`) applied here in date order with the earlier recommendations of the run; what it catches is taken
off the order and kept as a hypothetical outcome. A live Advisor, which ranks WITH its ledger, moves on to the next company; so PHASE 1 asks it for `time_machine.spare_recommendations` MORE than the number asked for (never past the Advisor's own limit) and the walk
takes them in rank order, skips what the cooldown catches and keeps the first `find` (the number a live Advisor would have returned): the ones beyond it are `not_traded_reason: "beyond_count"`, kept as hypothetical outcomes like the cooled-down ones. The Critic
has reviewed all of them in phase 1, so the walk still calls no model. A day trades fewer than the number asked for only when the cooldown leaves fewer than `find` (Critic v2 spec section 8, design choices entry 135).
One exit rule: after the walk every filled trade is checked against `Trade.outcome` (the Trader's `follow`) and the sold position against `SimPortfolio.step`: the same date, price and reason, or the run fails loudly (`ExitRuleError`).
(A NEUTRAL company's purchase is the exception: its ticket has no plan, so the Trader's follower never closes it, and the account follows it by the Advisor's EXIT plan (Advisor config version 2: a stop and a time stop, no target); both use `simulator.follow`, and the walk says so in its notes.)

NEVER THE FUTURE (P2). Every reader is bound to the day; the walk's own reads of bars after a decision (the fill, the follower) are the Trader's and `step`'s explicit ranges, cut at `bars.usable_last_day` (nothing dated today is a close).
NEVER A FETCH, NEVER A WRITE OUTSIDE THE STORE (P6): no fetch, the account is in memory, the ledgers are throwaway and the results are files under `data/proving/`. `PAPER_TRADING` off refuses the run.
MONEY is Decimal through the doors (P7); a return is a measurement. A failed day does not halt a run (the Trader's own rule, entry 82); `should_stop` is checked between days and a stopped run is saved as `stopped` with what it has.
"""
import concurrent.futures
import contextvars
import copy
import hashlib
import json
import tempfile
import threading
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Annotated, Any, Callable, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, StrictBool, field_validator

from prometheus.agents.trader.config import load_config as load_trader_config
from prometheus.common import money as money_module
from prometheus.common import shares as sh
from prometheus.common import trace, trading_clock
from prometheus.common.market_calendar import is_trading_day, load_calendar, next_trading_day, simulated_moment
from prometheus.common.money import PositiveMoney, dumps
from prometheus.common.portfolio import Portfolio, trade_amount
from prometheus.common.user_account import parse_money
from prometheus.common.work_order import MAX_TOP_K, TRADING, WorkOrder
from prometheus.proving import bars as bars_door
from prometheus.proving import coverage as coverage_module
from prometheus.proving import outcomes as outcomes_module
from prometheus.proving import recs, record, splits as splits_mod, stats as stats_mod, track
from prometheus.proving import speed as speed_module
from prometheus.proving import tracing as proving_tracing
from prometheus.proving.config import ProvingConfig, load_config
from prometheus.proving.memo import Memo
from prometheus.proving.portfolio import LotInfo, SimPortfolio
from prometheus.proving.record import CAVEATS, DecisionDay, EquityPoint, Event, LangSmith, ProvingRun, RecOutcome, RunRange
from prometheus.proving.store import ProvingStore
from prometheus.rag.config import REPO_ROOT, setting

Emit = Callable[[str, dict], None]


class RefusedRequest(ValueError):
    """The request cannot be run, and why, in words for the user (the usable range is part of the message)."""


class ExitRuleError(AssertionError):
    """The Trader's follower and the account's step disagree about an exit: there must be one exit rule (spec 7.2)."""


def _cash(value):
    return parse_money(value) if isinstance(value, str) else value


class TimeMachineRequest(BaseModel):
    """What the user asks of a run (spec 7.1). `cash` is the test's own starting money (a string of dollars or a number: the door of money takes it by its digits)."""
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=600)
    cash: Annotated[PositiveMoney, BeforeValidator(_cash)]
    start: date
    end: date
    cadence: str | None = None
    risk: Literal["conservative", "moderate", "aggressive"] = "moderate"
    neutral: StrictBool = False
    strict: StrictBool = False
    speed: Literal["fast", "faithful"] | None = None      # how the models are asked (design/proving_ground_spec.md P14): None is the config's `time_machine.mode` (fast); "faithful" asks exactly what the live desk asks and takes longer

    @field_validator("prompt")
    @classmethod
    def _clean(cls, value: str) -> str:
        return " ".join(value.split())


def engine_settings(config: ProvingConfig | None = None) -> dict:
    """The pool's settings from `configs/proving_v1.json` (`time_machine`): `parallel_days` (the most days researched at once), `retry_429` and `pause_429_seconds` (a day a provider's rate limit or a bad reply stopped is tried again, with a growing pause)."""
    machine = (config or load_config()).time_machine
    return {"parallel_days": machine.parallel_days, "retry_429": machine.retry_429, "pause_429_seconds": machine.pause_429_seconds}


# ---------------------------------------------------------------- the plan: decision days, coverage, the cache

def decision_days(start: date, end: date, cadence: str, config: ProvingConfig | None = None) -> list[date]:
    """The trading days of [start, end] the cadence picks (a whole number N: every Nth trading day from the start; a list of weekdays: those days of each week, the next trading day of the same week when one is a holiday). Raises
    ValueError for a cadence that is not in the config."""
    config = config or load_config()
    cadences = config.time_machine.cadences
    if cadence not in cadences:
        raise ValueError(f"the cadence {cadence!r} is not one of {', '.join(cadences)}")
    calendar = load_calendar()
    trading = []
    day = start
    while day <= end:
        if is_trading_day(day, calendar):
            trading.append(day)
        day += timedelta(days=1)
    spec = cadences[cadence]
    if isinstance(spec, int):
        return trading[::spec]
    picked: set[date] = set()
    monday = start - timedelta(days=start.weekday())
    while monday <= end:
        week_end = monday + timedelta(days=4)
        for weekday in spec:
            wanted = monday + timedelta(days=weekday)
            while wanted <= week_end and not is_trading_day(wanted, calendar):
                wanted += timedelta(days=1)
            if wanted <= week_end and start <= wanted <= end:
                picked.add(wanted)
        monday += timedelta(days=7)
    return sorted(picked)


def versions() -> dict:
    """The config versions that make the research what it is (the Advisor's, the Critic's, the Account Manager's, the Trader's and the Proving Ground's)."""
    out = {}
    for name in ("advisor_v1", "critic_v1", "account_manager_v1", "trader_v1", "proving_v1"):
        try:
            out[name] = json.loads((REPO_ROOT / "configs" / f"{name}.json").read_text(encoding="utf-8")).get("config_version")
        except (OSError, ValueError):
            out[name] = None
    return out


def models_used() -> dict:
    from prometheus.models.chat import model_name
    return {role: model_name(role) for role in ("data_clerk", "advisor", "critic", "account_manager", "guard")} | {"trader": None}


def data_stamp(report: dict) -> str:
    """What the stored data was when the research was made: the last bar day and the counts of the stored text per month, so that a backfill or a refresh makes a different key."""
    keep = {m: {k: row.get(k) for k in ("bars", "news", "press_releases", "filings", "news_chunks", "press_release_chunks", "filing_chunks")} for m, row in report.get("months", {}).items()}
    return hashlib.sha256(json.dumps({"last_bar_day": report.get("last_bar_day"), "months": keep}, sort_keys=True).encode()).hexdigest()[:16]


def spare_of(config: ProvingConfig | None = None) -> int:
    """How many recommendations MORE than the number asked for phase 1 asks the Advisor for (`time_machine.spare_recommendations`, Critic v2 spec section 8). A research made before the spare existed asked for none: a test of the real v1 research says so."""
    return (config or load_config()).time_machine.spare_recommendations


def research_key(intent, day: date, request: TimeMachineRequest, stamp: str, config: ProvingConfig | None = None) -> str:
    """The key of one day's research (spec 7.4): sha256 of the intent as the Account Manager read it, the day, the risk, the two switches, the config versions, the models and the data's stamp. THE SPEED is part of it, so a fast
    day never answers a faithful run: a FAST key also holds the fast settings (a changed effort or cap is another research); a FAITHFUL key is exactly the key made before the speeds existed (the faithful settings ARE the live desk's, and
    the real captures of `tests/fixtures/proving_real/` were made with them), so every faithful capture still answers a faithful run."""
    config = config or load_config()
    body = {"intent": intent.model_dump(mode="json"), "day": day.isoformat(), "risk": request.risk, "neutral": request.neutral, "strict": request.strict, "versions": versions(), "models": models_used(), "stamp": stamp}
    if spare_of(config):      # the Advisor was asked for more than the number asked: another research (a key made with none is exactly the key made before the spare existed)
        body["spare"] = spare_of(config)
    speed = speed_module.speed_of(request.speed, config)
    if speed == "fast":
        body["speed"], body["fast"] = speed, speed_module.mode_summary(speed, config)
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:40]


def check(request: TimeMachineRequest, report: dict, config: ProvingConfig | None = None) -> list[str]:
    """The reasons a request cannot run (empty: it can), each with the usable range (spec 7.1). Nothing is paid for before this is empty."""
    config = config or load_config()
    problems = []
    usable = report["usable"]
    first, last = usable.get("first_day"), usable.get("last_day")
    where = f"the usable range is {first} to {last}"
    calendar = load_calendar()
    cadence = request.cadence or config.time_machine.default_cadence
    if cadence not in config.time_machine.cadences:
        problems.append(f"the cadence {cadence!r} is not one of {', '.join(config.time_machine.cadences)}")
    for label, day in (("start", request.start), ("end", request.end)):
        if not is_trading_day(day, calendar):
            problems.append(f"the {label} day {day} is not a trading day ({where})")
    if not request.start < request.end:
        problems.append(f"the start {request.start} must be before the end {request.end} ({where})")
    if first and request.start < date.fromisoformat(first):
        problems.append(f"the start {request.start} is before the first day with enough stored history ({where}: {config.time_machine.min_history_sessions} stored sessions of bars and {config.time_machine.min_news_days} days of news come before it)")
    if last and request.end > date.fromisoformat(last):
        problems.append(f"the end {request.end} is after the last stored bar that may be read as a close ({where})")
    if not problems:
        count = len(decision_days(request.start, request.end, cadence, config))
        if count == 0:
            problems.append(f"the cadence {cadence} has no decision day between {request.start} and {request.end}")
        if count > config.time_machine.max_decision_days:
            problems.append(f"{count} decision days is more than the {config.time_machine.max_decision_days} a run may have: shorten the range or take a slower cadence")
    return problems


def reading_key(request: TimeMachineRequest) -> str:
    """The key of the stored reading of a request: the prompt, the Account Manager's config version and its model."""
    body = {"prompt": request.prompt, "version": versions().get("account_manager_v1"), "model": models_used().get("account_manager")}
    return "reading-" + hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:32]


def stored_reading(request: TimeMachineRequest, store: ProvingStore):
    """The `TradeIntent` the Account Manager read from this prompt in an earlier run (kept in the cache folder), or None. A run that finds one does not call the model again: the same words are read the same way, and the research cache hits."""
    from prometheus.agents.account_manager.intent import TradeIntent
    kept = store.cache_get(reading_key(request))
    try:
        return TradeIntent.model_validate(kept["intent"]) if kept else None
    except (KeyError, ValueError):
        return None


def plan(request: TimeMachineRequest, store: ProvingStore | None = None, report: dict | None = None, intent=None) -> dict:
    """What a run would be, free of any model call (the page's `POST /api/proving/plan`): `problems` (empty when it can run), the decision days, how many are in the cache (known only when `intent` is given: the cache key holds the
    way the request was read), the caveats that will apply, the cost estimate (the config's, or None), the `speed` of the run and its `timing`: {speed, parallel_days (how many days are researched at once), cached_days, paid_days,
    estimate_seconds (the config's measured wall time of a cold run of this speed, 0 when every day is cached, None when none was measured)}."""
    config, store = load_config(), store or ProvingStore()
    report = report or coverage_module.compute()
    intent = intent or stored_reading(request, store)
    problems = check(request, report, config)
    cadence = request.cadence or config.time_machine.default_cadence
    speed = speed_module.speed_of(request.speed, config)
    days = [] if problems else decision_days(request.start, request.end, cadence, config)
    stamp = data_stamp(report)
    cached = [d for d in days if intent is not None and store.has_cache(research_key(intent, d, request, stamp, config))]
    codes = ["fundamentals_not_point_in_time", "symbol_directory_is_todays", "price_returns_no_dividends", "fills_at_next_open_no_slippage", "daily_bars_stop_before_target", "open_positions_marked", "model_knowledge_cutoff"]
    if speed == "fast":
        codes.append("fast_models")
    mode = config.time_machine.fast if speed == "fast" else config.time_machine.faithful
    note = mode.cost_note_per_day_usd or config.time_machine.cost_note_per_day_usd      # the measured cost of a day in THIS mode, else the section's own
    paid = (len(days) - len(cached)) if intent is not None else len(days)
    noted = mode.run_seconds_note
    timing = {"speed": speed, "parallel_days": min(config.time_machine.parallel_days, max(paid, 1)), "cached_days": len(cached), "paid_days": paid, "estimate_seconds": 0 if days and paid == 0 else noted}
    return {"problems": problems, "cadence": cadence, "decision_days": [d.isoformat() for d in days], "cached_days": [d.isoformat() for d in cached], "cache_known": intent is not None, "paid_days": paid,
            "caveats": [{"code": c, "sentence": CAVEATS[c]} for c in codes], "cost_estimate_usd": None if note is None else note * paid, "range": report["usable"], "last_bar_day": report["last_bar_day"], "speed": speed, "timing": timing}


# ---------------------------------------------------------------- phase 1: one day's research

class Day:
    """What phase 1 gives for one decision day."""
    def __init__(self, day: date, order: dict | None, plan: dict | None, status: str, seconds: float, cached: bool, reason: str | None = None, usage: dict | None = None, trace_id: str | None = None, timing: dict | None = None,
                 news_sources: int | None = None):
        self.day, self.order, self.plan, self.status, self.seconds, self.cached, self.reason, self.usage, self.trace_id, self.timing = day, order, plan, status, seconds, cached, reason, usage, trace_id, timing
        self.news_sources = news_sources      # how many news articles the research found for the day's companies (None: a research made before it was counted)


def _profile(request: TimeMachineRequest):
    from prometheus.common.user_profile import UserProfile
    return UserProfile(user="Proving Ground", risk_tolerance=request.risk, trade_mode="auto")


def _throwaway_ledger():
    from prometheus.memory.ledger import Ledger
    return Ledger(Path(tempfile.mkdtemp(prefix="prometheus_proving_ledger_")) / "recommendations.jsonl")


def _rate_limited(order: WorkOrder | None) -> bool:
    text = " ".join(str(e) for e in [*(order.errors if order else []), *[r.get("error") for r in (order.advisor_runs if order else []) + (order.critic_runs if order else []) if isinstance(r, dict)]] if e).lower()
    return "429" in text or "rate limit" in text or "ratelimit" in text


def _stage_of(state: dict) -> bool:
    """Whether the desk has gone as far as the research goes: the Critic has reviewed, or the Advisor recommended nothing for it to review."""
    order, path = state.get("order"), state.get("path") or []
    if "critic" in path:
        return True
    return "advisor" in path and order is not None and not order.recommendations


class SpareAdvisor:
    """The Advisor of phase 1, asked for `spare` more recommendations than the desk asks it for (Critic v2 spec section 8): everything else is the Advisor's own (a thin proxy: `generate_recommendations` with a larger `top_k`, never past
    `work_order.MAX_TOP_K`, the limit the Advisor itself keeps; every other attribute is the Advisor's). The walk keeps the first `find` of them that the Advisor's cooldown does not catch, so a day with a cooled-down company in its first places still
    trades the number asked for, as a live Advisor that ranks with its ledger would."""

    def __init__(self, advisor, spare: int):
        self.advisor, self.spare = advisor, spare

    def generate_recommendations(self, order, case=None, today=None, top_k=None, only_subject=False):
        asked = top_k if top_k is None else min(top_k + self.spare, MAX_TOP_K)
        return self.advisor.generate_recommendations(order, case, today, top_k=asked, only_subject=only_subject)

    def __getattr__(self, name):
        return getattr(self.advisor, name)


def research_day(day: date, request: TimeMachineRequest, intent, memo: Memo, guard, run_id: str, audit_path: Path, models: speed_module.Models | None = None) -> Day:
    """PHASE 1 for one day, in the calling thread (the engine runs it inside a copy of its context): the real `build_desk` as of `day`, stopped when the Critic has reviewed. Never raises: a failure is a `failed` day with its reason.
    `models` is how the run's speed asks the models (`speed.build_models`): left out, or faithful, the agents build their own exactly as the live desk does; in the fast mode the Data Clerk, the Advisor and the Critic are handed the models
    that section of the config describes, the Advisor's calls wait at most its call limit, and the Critic checks the records of a review at once."""
    from prometheus.agents.account_manager.account_manager import AccountManager
    from prometheus.agents.advisor.advisor import Advisor
    from prometheus.agents.critic.critic import Critic
    from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
    from prometheus.agents.data_clerk.data_toolbox import DataToolbox
    from prometheus.langgraph.desk import build_desk
    from prometheus.tools import symbols
    models = models or speed_module.Models(speed="faithful")
    began = time.perf_counter()
    usage = proving_tracing.UsageHandler()
    stages: dict[str, float] = {}

    def done(status: str, order, plan_, reason: str | None) -> Day:
        """The day's result with its measurement: the seconds of each desk node and every model call (model, stage, start, seconds, tokens)."""
        seconds = time.perf_counter() - began
        return Day(day, order, plan_, status, seconds, False, reason, usage.summary(), trace_id, {"seconds": round(seconds, 2), "stages": dict(stages), "calls": usage.call_rows()})
    with trace.span("proving.decision_day", metadata={"as_of": day.isoformat(), "cached": False, "proving_run_id": run_id}) as day_span:
        trace_id = str(day_span.id) if day_span is not None else None
        try:
            with trading_clock.as_of(day), speed_module.limit_of(models.call_limit):
                profile = _profile(request)
                own = lambda role: {} if getattr(models, role) is None else {"model": getattr(models, role)}      # a model the speed built; otherwise the agent builds its own, as the live desk does
                clerk = DataClerk(allow_fetch=False, toolbox=DataToolbox(base_dir=REPO_ROOT, memo=memo), symbol_search=memo.wrap("symbol_search", symbols.search_finnhub), guard=guard, **own("data_clerk"))
                ledger = _throwaway_ledger()
                advisor = SpareAdvisor(Advisor(clerk=clerk, ledger=ledger, **own("advisor")), spare_of())
                critic = Critic(clerk=clerk, ledger=_throwaway_ledger(), history=ledger, parallel=models.parallel_records, **own("critic"))

                def manager():
                    made = AccountManager(model=object(), clerk=clerk, profile=profile)
                    made.understand_request = lambda text: intent      # the request was read ONCE by the Account Manager (the memo): every day works from that reading
                    return made
                desk = build_desk(clerk_factory=lambda order: clerk, advisor_factory=lambda order: advisor, funds_factory=manager, with_critic=True, critic_factory=lambda order: critic,
                                  clock=lambda: simulated_moment(day), guard=guard)
                portfolio = Portfolio(user=profile.user, cash_avail=request.cash)
                state0 = {"request": {"ask": {"text": request.prompt, "default_count": 3, "fetch": False}, "account": {"profile": profile, "portfolio": portfolio, "mode": "auto"}}, "order": None, "errors": [], "resolution": None,
                          "funds": None, "path": [], "seconds": {}, "with_trader": False, "plan": None}
                config = {"callbacks": [usage], "metadata": {"as_of": day.isoformat(), "proving_run_id": run_id}, "tags": ["proving"]}
                final = None
                mark = time.perf_counter()
                for state in desk.stream(state0, stream_mode="values", config=config):
                    final = state
                    node = (state.get("path") or [None])[-1]
                    if node:      # the time since the last node finished is this node's (the nodes of a day run one after the other)
                        stages[node] = round(time.perf_counter() - mark, 2)
                        mark = time.perf_counter()
                    if _stage_of(state):
                        break
            order, plan_ = (final or {}).get("order"), (final or {}).get("plan")
            if order is None or plan_ is None:
                reason = "; ".join((final or {}).get("errors") or []) or "the desk did not start"
                return done("failed", None, None, reason)
            if order.status == "data_failed" or (order.errors and not order.recommendations and not order.candidates):
                return done("failed", None, None, "; ".join(order.errors)[:300] or "the data was not ready")
            if _rate_limited(order):
                return done("failed", None, None, "a provider's rate limit (429) stopped the research of the day")
            broken = [str(r.get("error") or r.get("errors")) for r in [*order.advisor_runs, *order.critic_runs] if isinstance(r, dict) and (r.get("error") or r.get("errors"))]
            if broken:      # a model that failed (a reply that was not valid, a timeout, a model that did not start) leaves a day with NO recommendations that a live run would have made: that is a failed day, never a cached empty one
                return done("failed", None, None, "a model call failed: " + "; ".join(dict.fromkeys(broken))[:200])
            stored = json.loads(order.model_dump_json(exclude={"data"}))
            stored["data"] = {"companies": order.data.companies if order.data else [], "notes": list(order.data.notes) if order.data else []}      # the articles are left out; what the decision reads stays
            scalars = {k: plan_.get(k) for k in ("balance", "budget", "each", "count", "said_how_much", "request_text", "find", "case", "action", "max_price")}
            scalars.update({k: plan_[k] for k in ("min_price", "shares_of", "shares_each") if plan_.get(k)})      # a price floor and the shares typed with a recommendation (entry 140): kept only when the request has them, so every earlier saved day is read as before
            finished = done("done", stored, json.loads(dumps(scalars)), None)
            try:
                finished.news_sources = len(order.data.sources_of("news")) if order.data is not None and order.data.results else None
            except Exception:
                pass
            return finished
        except Exception as error:      # a model error, a refusal, a store that could not be read: the day is failed and the walk goes on
            return done("failed", None, None, f"{type(error).__name__}: {' '.join(str(error).split())[:200]}")


def warm_stores() -> None:
    """Open the stores of stored text once, in the calling thread (the embedder and the vector store's client are loaded here): the pool of days then shares them, and 24 threads never build the embedder at the same time. Never raises."""
    try:
        from prometheus.rag.store import get_vector_store
        for kind in ("news", "press_releases", "filings"):
            get_vector_store(kind)
    except Exception:
        pass


def research_all(days: list[date], request: TimeMachineRequest, intent, store: ProvingStore, use_cache: bool, stamp: str, memo: Memo, guard, run_id: str, emit: Emit, should_stop, settings: dict,
                 models_factory: Callable[[], speed_module.Models] | None = None, config: ProvingConfig | None = None, warming: threading.Thread | None = None) -> dict[date, Day]:
    """PHASE 1 for every decision day: the cached ones at once, the others ALL AT ONCE in a pool of at most `parallel_days` threads (each in its own copy of this context), results emitted as they finish. `models_factory` makes the
    run's models (`speed.build_models`) only when a day has to be researched: a run that is all cache needs no key and no model. A day a provider's rate limit (429) or a bad reply stopped is tried again (`retry_429`, with a growing pause)."""
    config = config or load_config()
    results: dict[date, Day] = {}
    keys = {d: research_key(intent, d, request, stamp, config) for d in days}
    speed = speed_module.speed_of(request.speed, config)
    total, done = len(days), 0
    todo = []
    for d in days:
        hit = store.cache_get(keys[d]) if use_cache else None
        if hit is not None and isinstance(hit.get("order"), dict):
            results[d] = Day(d, hit["order"], hit.get("plan") or {}, "done", 0.0, True, None, hit.get("usage"), None, {"seconds": 0.0, "cached": True, "researched_in": (hit.get("timing") or {}).get("seconds")}, hit.get("news_sources"))
            done += 1
            with trace.span("proving.decision_day", metadata={"as_of": d.isoformat(), "cached": True, "proving_run_id": run_id}) as span:
                results[d].trace_id = str(span.id) if span is not None else None
            emit("pv_research", _research_event(results[d], done, total))
        else:
            todo.append(d)
    if not todo:
        return results      # all cache: nothing is loaded, no model is built, no key is read
    if warming is not None:
        warming.join()
    else:
        warm_stores()
    try:
        models = (models_factory or (lambda: speed_module.Models(speed="faithful")))()
    except (RuntimeError, ValueError) as error:      # for example a missing key: every day that needed a model is failed with the reason (the message names the setting, never its value)
        for d in todo:
            done += 1
            results[d] = Day(d, None, None, "failed", 0.0, False, f"{type(error).__name__}: {' '.join(str(error).split())[:200]}", None, None, {"seconds": 0.0})
            emit("pv_research", _research_event(results[d], done, total))
        return results
    audit_path = store.root / "audit" / f"{run_id}.jsonl"
    pool_size = max(1, min(settings["parallel_days"], len(todo)))
    limit = threading.Semaphore(pool_size)
    lock = threading.Lock()
    pause = settings.get("pause_429_seconds", 5.0)

    def task(d: date) -> Day:
        tried: list[dict] = []      # what each attempt at this day was: the seconds a failed one took are part of the run's time, and its reason is never lost
        for attempt in range(settings["retry_429"] + 1):
            with limit:
                if should_stop():
                    return Day(d, None, None, "failed", 0.0, False, "stopped before this day was researched", None, None)
                found = research_day(d, request, intent, memo, guard, run_id, audit_path, models)
            tried.append({"status": found.status, "seconds": round(found.seconds, 2), "reason": found.reason})
            if found.timing is not None:
                found.timing["attempts"] = attempt + 1
                found.timing["attempt_log"] = list(tried)
            retryable = "429" in (found.reason or "") or (found.reason or "").startswith("a model call failed")      # a rate limit, or a model that answered badly once: the day is tried again (never cached while it fails)
            if not retryable or attempt >= min(settings["retry_429"], 2):
                return found
            time.sleep(pause * (attempt + 1))      # a provider's rate limit: the pool waits, and tries the day again with a growing pause
        return found
    with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as pool:
        futures = {pool.submit(contextvars.copy_context().run, task, d): d for d in todo}
        for future in concurrent.futures.as_completed(futures):
            found = future.result()
            with lock:
                results[found.day] = found
                done += 1
                if found.status == "done" and use_cache:
                    store.cache_put(keys[found.day], {"order": found.order, "plan": found.plan, "usage": found.usage, "timing": found.timing, "news_sources": found.news_sources, "speed": speed, "day": found.day.isoformat(),
                                                      "stored_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
                emit("pv_research", _research_event(found, done, total))
    return results


def _research_event(day: Day, done: int, total: int) -> dict:
    order = day.order or {}
    recs_ = order.get("recommendations") or []
    reviews = order.get("reviews") or []
    return {"day": day.day.isoformat(), "status": "cached" if day.cached and day.status == "done" else day.status, "cached": day.cached, "recommended": len(recs_), "approved": sum(1 for r in reviews if r.get("approved") is True),
            "denied": sum(1 for r in reviews if r.get("approved") is not True), "seconds": round(day.seconds, 1), "reason": day.reason, "done": done, "total": total, "usage": day.usage,
            "stages": (day.timing or {}).get("stages") or {}}


# ---------------------------------------------------------------- phase 2: the walk

def _clamped_manager(clerk, profile):
    """The Account Manager of the walk: the real one, whose `request_bars` is cut at the last usable bar (nothing dated today is a close: the Trader's follower and the account's step read the same bars)."""
    from prometheus.agents.account_manager.account_manager import AccountManager

    class ProvingManager(AccountManager):
        def request_bars(self, symbols, start, end):
            return super().request_bars(symbols, start, min(end, bars_door.usable_last_day()))
    return ProvingManager(model=object(), clerk=clerk, profile=profile)


def cooldown_filter(order: WorkOrder, day: date, earlier: list[dict]) -> list[dict]:
    """Take off `order` (in place) the recommendations the Advisor's own `ranking.cooldown_of` would have hidden: an earlier recommendation of this run, same symbol and family, whose time stop has not passed and none of whose
    invalidation rules fired as of `day`. Returns the recommendations that were taken off (their reviews too)."""
    from types import SimpleNamespace

    from prometheus.agents.advisor import ranking
    caught = []
    for rec in list(order.recommendations):
        symbol, family = rec.get("symbol"), (rec.get("trade") or {}).get("family")
        seen = SimpleNamespace(metrics=rec.get("features") or {}, events=(rec.get("evidence") or {}).get("events") or [])      # the Advisor's own record of the day: the metrics and events its invalidation rules read
        old = ranking.cooldown_of(seen, family, [r for r in earlier if r.get("symbol") == symbol], day.isoformat()) if family else None
        if old:
            caught.append({**rec, "_cooldown_of": old})
            order.recommendations.remove(rec)
            record_id = (rec.get("identity") or {}).get("record_id")
            order.reviews = [r for r in order.reviews if r.get("record_id") != record_id]
    return caught


def beyond_count(order: WorkOrder, find) -> list[dict]:
    """Take off `order` (in place) the recommendations after the first `find` (the Advisor's rank order, after the cooldown took its own): what a live Advisor, asked for `find`, would not have returned (Critic v2 spec section 8; phase 1 asked for
    `spare` more). Returns the recommendations that were taken off (their reviews too); nothing when `find` is not a whole number above zero (a research that did not say it keeps every recommendation it has)."""
    if not isinstance(find, int) or isinstance(find, bool) or find < 1:
        return []
    extra = list(order.recommendations[find:])
    order.recommendations = list(order.recommendations[:find])
    gone = {(r.get("identity") or {}).get("record_id") for r in extra}
    order.reviews = [r for r in order.reviews if r.get("record_id") not in gone]
    return extra


def check_exits(trades: list[dict], portfolio: SimPortfolio, splits: dict, walked_until: date | None = None) -> list[str]:
    """One exit rule (spec 7.2): each filled purchase's `Trade.outcome` (the Trader's follower) against the account's own sale or open lot. Raises ExitRuleError on a disagreement; a symbol with a split table or a suspected split is
    followed by the account's own split rules and is left out (said in the returned notes). `walked_until` is the last day the walk reached (a stopped run did not walk to the end): an exit after it has not happened yet and is not compared."""
    from prometheus.agents.trader.record import Trade
    notes = []
    sold = {(e.record_id, e.symbol): e for e in portfolio.closed}
    held = {(lot.record_id, lot.symbol) for lot in portfolio.lots}
    neutral = 0
    for raw in trades:
        trade = Trade.model_validate(raw)
        if trade.status != "filled" or trade.side != "buy" or trade.outcome is None:
            continue
        if (trade.record_id or "").startswith("neutral-"):      # a NEUTRAL company: the Trader's ticket has no plan (market order), so its follower never closes it; the account follows it by the Advisor's exit plan (Advisor v2): nothing to compare
            neutral += 1
            continue
        key = (trade.record_id, trade.symbol)
        table = splits.get(trade.symbol)
        crossed = table is not None and any(split.execution_date >= date.fromisoformat(trade.fill.date) for split in table.splits)      # a split on or after the day of the fill: the account follows it with its own split rules
        if crossed or any(trade.symbol in w for w in portfolio.warnings):
            notes.append(f"{trade.symbol}: followed with its split rules, not compared with the Trader's follower")
            continue
        out = trade.outcome
        if out.status == "closed" and walked_until is not None and date.fromisoformat(out.exit_date) > walked_until:
            continue
        if out.status == "closed":
            exit = sold.get(key)
            if exit is None or exit.date.isoformat() != out.exit_date or exit.price != out.exit_price or exit.reason != out.exit_reason:
                raise ExitRuleError(f"{trade.symbol}: the Trader's follower says closed on {out.exit_date} at {out.exit_price} ({out.exit_reason}), the account says {None if exit is None else (exit.date.isoformat(), exit.price, exit.reason)}")
        elif key not in held:
            raise ExitRuleError(f"{trade.symbol}: the Trader's follower says the position is still open, the account says it was sold")
    if neutral:
        notes.append(f"{neutral} neutral purchase(s) followed by the Advisor's exit plan (a stop and a time stop), not compared with the Trader's follower: its ticket has no plan for a neutral company")
    return notes


def _benchmark_curve(points: list[EquityPoint], spy: list[dict], cash: Decimal, first_day: date) -> list[EquityPoint]:
    """The benchmark's money: the same starting cash put into SPY (fractional, six places down) at the open of `first_day`, marked at each close; before that day it is the cash. Each point gets its `benchmark`."""
    from prometheus.agents.trader import simulator
    from prometheus.common import shares as sh
    bars = {b["date"]: b for b in spy if isinstance(b, dict) and isinstance(b.get("date"), str)}
    opened = simulator.bar_price(bars.get(first_day.isoformat()) or {}, "open")
    if opened is None:
        return points
    count = sh.quantity((cash / opened).quantize(sh.STEP, rounding=ROUND_DOWN))
    left = cash - trade_amount(count, opened)
    out = []
    for p in points:
        close = simulator.bar_price(bars.get(p.date.isoformat()) or {}, "close") if p.date >= first_day else None
        out.append(p.model_copy(update={"benchmark": left + trade_amount(count, close) if close is not None else (cash if p.date < first_day else None)}))
    last = None
    for i, p in enumerate(out):      # a day with no SPY bar keeps the last value (the benchmark is never invented, but a gap in a line is a mark)
        if p.benchmark is None and last is not None:
            out[i] = p.model_copy(update={"benchmark": last})
        last = out[i].benchmark if out[i].benchmark is not None else last
    return out


def run_time_machine(request: TimeMachineRequest, *, store: ProvingStore | None = None, emit: Emit | None = None, should_stop: Callable[[], bool] | None = None, use_cache: bool = True, intent=None, guard=None, config: ProvingConfig | None = None,
                     run_id: str | None = None) -> ProvingRun:
    """Run a Time Machine test (see the module text) and return the saved `ProvingRun`. `intent` is the request as the Account Manager read it (read ONCE with its model when left out: a model call, and the security layer's
    screening of the request); `emit(type, data)` gets the `pv_*` events; `should_stop()` is asked between days. Raises RefusedRequest before anything is paid for; a failed decision day is recorded and the run goes on."""
    from prometheus.agents.trader.config import paper_trading_enabled
    from prometheus.agents.trader.trader import NullLedger, Trader
    from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
    from prometheus.application.web import settings as run_settings_module
    from prometheus.security.audit import AuditLog
    from prometheus.security.guard import Guard

    began = time.perf_counter()
    config, store, emit = config or load_config(), store or ProvingStore(), emit or (lambda type_, data: None)
    should_stop = should_stop or (lambda: False)
    settings = engine_settings()
    speed = speed_module.speed_of(request.speed, config)
    if not paper_trading_enabled():
        raise RefusedRequest("PAPER_TRADING is off: the Proving Ground runs simulated brokers only, and refuses to run when paper trading is not on")
    report = coverage_module.compute()
    problems = check(request, report, config)
    if problems:
        raise RefusedRequest("; ".join(problems))
    cadence = request.cadence or config.time_machine.default_cadence
    days = decision_days(request.start, request.end, cadence, config)
    run_id = run_id or f"tm-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
    memo = Memo()
    chosen = run_settings_module.run_settings(request.strict, request.neutral)
    understood_cost = {"calls": 0, "tokens_in": 0, "tokens_out": 0}
    # ---- the request is read ONCE, and screened once (as a live run does)
    if intent is None and use_cache:
        intent = stored_reading(request, store)      # the same words were read in an earlier run: the same reading, no model call
    warming = None
    if intent is None:      # a cold run: the stores are opened while the request is screened and read (two model calls); a run that is all cache never opens them
        warming = threading.Thread(target=contextvars.copy_context().run, args=(warm_stores,), name="proving-warm-stores", daemon=True)
        warming.start()
    if intent is None:
        from prometheus.agents.account_manager.account_manager import AccountManager
        screening = Guard.with_scorer().screen_request(request.prompt, rate=False)
        if screening.refused:
            raise RefusedRequest(screening.message)
        reader = AccountManager(profile=_profile(request))
        intent = memo.get("request_reading", request.prompt, lambda: reader.understand_request(request.prompt))
        used = getattr(reader, "intent_run", None) or {}
        understood_cost = {"calls": used.get("calls", 0), "tokens_in": used.get("tokens_in", 0), "tokens_out": used.get("tokens_out", 0)}
        if use_cache and intent.action != "order":
            store.cache_put(reading_key(request), {"intent": intent.model_dump(mode="json"), "prompt": request.prompt})
    if intent.action == "order":
        raise RefusedRequest("a direct order names the trade itself, so there is no recommendation in it to test: ask for recommendations (a theme, a list of companies or a company)")
    guard = (guard or Guard.code_only(audit=AuditLog(store.root / "audit" / f"{run_id}.jsonl"))).without_rate_limit()
    stamp = data_stamp(report)
    trader_config = load_trader_config()
    with run_settings_module.applied(chosen), trace.root("proving.time_machine", metadata={"proving_run_id": run_id, "kind": "time_machine", "start": request.start.isoformat(), "end": request.end.isoformat(), "cadence": cadence,
                                                                                         "decision_days": len(days), "cash": float(request.cash), "risk": request.risk, "neutral": request.neutral, "strict": request.strict, "speed": speed}, tags=["proving", "time_machine", speed],
                                                                         inputs={"request": request.prompt}) as root:
        emit("pv_started", {"run_id": run_id, "request": request.model_dump(mode="json"), "understood": intent.describe(), "range": {"first_day": days[0].isoformat(), "last_day": days[-1].isoformat(), "last_bar_day": report["last_bar_day"]},
                            "decision_days": [d.isoformat() for d in days], "starting_cash": money_module.json_number(request.cash),
                            "caveats": [{"code": c, "sentence": CAVEATS[c]} for c in track.caveats_of([], config, simulated_fills=True, models_may_remember=True, not_point_in_time=True, fast_models=speed == "fast")],
                            "parallel_days": min(settings["parallel_days"], len(days)), "phase": "research", "speed": speed,
                            "timing": {"speed": speed, "parallel_days": min(settings["parallel_days"], len(days)), "estimate_seconds": (config.time_machine.fast if speed == "fast" else config.time_machine.faithful).run_seconds_note}})
        # ---- phase 1
        phase1 = time.perf_counter()
        setup_seconds = phase1 - began      # reading and screening the request (two model calls on a cold run), the coverage, the stores warmed up
        research = research_all(days, request, intent, store, use_cache, stamp, memo, guard, run_id, emit, should_stop, settings, lambda: speed_module.build_models(speed, config), config, warming)
        research_seconds = time.perf_counter() - phase1
        stopped = should_stop()
        # ---- phase 2
        phase2 = time.perf_counter()
        clerk = DataClerk(allow_fetch=False, model=object())      # the walk asks the Data Clerk for quotes and bars only (lookups, no model call): it needs no model, and no key
        profile = _profile(request)
        manager = _clamped_manager(clerk, profile)
        trader = Trader(mode="backtest", use_model=config.time_machine.use_trader_model, ledger=NullLedger())
        usable_last = bars_door.usable_last_day()
        wanted = {config.benchmark}
        for found in research.values():
            order_ = found.order or {}
            wanted |= {r.get("symbol") for r in order_.get("recommendations", []) + order_.get("candidates", []) if r.get("symbol")}
            wanted |= {((r.get("identity") or {}).get("sector_etf")) for r in order_.get("recommendations", []) if (r.get("identity") or {}).get("sector_etf")}
        book = bars_door.read_bars(sorted(wanted), request.start - timedelta(days=track.LOOKBACK_DAYS), usable_last, clerk=clerk)
        spy = book.get(config.benchmark, [])
        last_bar_day = date.fromisoformat(max((b["date"] for b in spy), default=request.end.isoformat()))
        walk_end = last_bar_day if config.time_machine.follow_after_end else min(request.end, last_bar_day)
        walk_days = sorted(date.fromisoformat(b["date"]) for b in spy if request.start.isoformat() <= b["date"] <= walk_end.isoformat())
        tables = {s: t for s in sorted(wanted) for t in [splits_mod.load(s)] if t is not None}
        account = SimPortfolio(request.cash, user=profile.user, splits=tables)
        decisions_on, pending, earlier = set(days), {}, []
        info: dict[str, LotInfo] = {}
        points, events, rows, all_outcomes, all_trades, due = [], [], [], [], [], {}
        desks: dict[str, dict] = {}
        first_fill: date | None = None
        notes: list[str] = []
        walk_span = trace.span("proving.walk", metadata={"days": len(walk_days)})
        walk_tree = walk_span.__enter__()
        for index, today in enumerate(walk_days):
            if should_stop():
                stopped = True
                break
            todays_events: list[Event] = []
            fills = pending.pop(today, [])
            if fills:
                for lot in account.apply_fills(fills, info):
                    first_fill = first_fill or today
                    todays_events.append(Event(date=today, kind="buy", symbol=lot.symbol, shares=lot.entry_shares, price=lot.entry_price, amount=lot.cost, key=lot.key))
            for exit_ in account.step(today, book):
                todays_events.append(Event(date=today, kind="exit", symbol=exit_.symbol, shares=exit_.shares, price=exit_.price, amount=exit_.amount, reason=exit_.reason, key=exit_.key))
            row = None
            if today in decisions_on and today <= request.end:
                found = research.get(today)
                row = DecisionDay(day=today, status="failed", reason=(found.reason if found else "no research") or "no research", trace_id=found.trace_id if found else None)
                if found is not None and found.status == "done":
                    try:
                        row, trades, scored, view = _decide_day(today, found, intent, request, manager, trader, account, earlier, book, config, trader_config, clerk, info)
                        all_trades.extend(trades)
                        for trade in trades:
                            if trade.get("status") == "filled" and trade.get("side") == "buy" and (trade.get("fill") or {}).get("date"):
                                pending.setdefault(date.fromisoformat(trade["fill"]["date"]), []).append(trade)
                        for outcome in scored:
                            all_outcomes.append(outcome)
                            due.setdefault(outcome.exit.date if outcome.exit else (outcome.path[-1].date if outcome.path else today), []).append(outcome)
                        desks[today.isoformat()] = view
                        emit("pv_decision", {"day": today.isoformat(), "order": view, "understood": intent.describe()})
                        row.trace_id = found.trace_id
                    except ExitRuleError:
                        raise
                    except Exception as error:      # a failed day does not halt a run
                        row = DecisionDay(day=today, status="failed", reason=f"{type(error).__name__}: {' '.join(str(error).split())[:200]}", trace_id=found.trace_id)
                rows.append(row)
            point = account.mark(today, book)
            if first_fill is not None:
                point = _benchmark_curve([point], spy, request.cash, first_fill)[0]
            else:
                point = point.model_copy(update={"benchmark": request.cash})
            points.append(point)
            events.extend(todays_events)
            emit("pv_day", {"point": point.model_dump(mode="json"), "events": [e.model_dump(mode="json") for e in todays_events], "decision_day": row.model_dump(mode="json") if row is not None else None, "index": index + 1, "total": len(walk_days)})
            for outcome in due.pop(today, []):
                emit("pv_outcome", {"outcome": outcome.model_dump(mode="json")})
        walk_span.__exit__(None, None, None)
        if not stopped:
            for left in sorted(due):
                for outcome in due[left]:
                    emit("pv_outcome", {"outcome": outcome.model_dump(mode="json")})
        if points:
            points = _benchmark_curve([p.model_copy(update={"benchmark": None}) for p in points], spy, request.cash, first_fill or walk_days[0])
        notes.extend(check_exits(all_trades, account, tables, points[-1].date if points else None))
        notes.extend(account.warnings)
        with trace.span("proving.stats") as stats_span:
            summary = stats_mod.compute(all_outcomes, points, config, starting_cash=request.cash)
            trace.set_outputs(stats_span, {"headline": summary.headline.sentence, "n": summary.outcomes.n})
        walk_seconds = time.perf_counter() - phase2
        usage = {d.isoformat(): research[d].usage for d in days if d in research and research[d].usage}
        langsmith, post_target, link_began = LangSmith(), {}, time.perf_counter()
        if root is not None:
            try:
                langsmith = LangSmith(trace_id=str(root.trace_id), url=root.get_url() if hasattr(root, "get_url") else None)
            except Exception:
                langsmith = LangSmith(trace_id=str(getattr(root, "trace_id", "")) or None)
            post_target = {"root_trace_id": str(root.trace_id), "address": getattr(root, "address", None)}      # where the day spans were sent: the feedback follows its run (posted below, after the result is out)
        link_seconds = time.perf_counter() - link_began
        trace.set_outputs(root, {"headline": summary.headline.sentence, "decision_days": len(days)})
    thin_days = [d.isoformat() for d in days if d in research and research[d].news_sources is not None and research[d].news_sources < config.time_machine.news_coverage_min_sources]
    analysts_used = any(isinstance(review.get("analysts"), dict) and review["analysts"].get("applicable") for d in days if d in research and research[d].order for review in (research[d].order.get("reviews") or []))       # a review scored the analysts' counts (Critic v3): they are each month's last revision, not what was known on the day
    caveats = track.caveats_of(all_outcomes, config, simulated_fills=True, thin_news=bool(thin_days), models_may_remember=True, not_point_in_time=True, fast_models=speed == "fast", analysts_used=analysts_used)
    cached_days = [d.isoformat() for d in days if d in research and research[d].cached]
    total_usage = {"calls": understood_cost["calls"], "tokens_in": understood_cost["tokens_in"], "tokens_out": understood_cost["tokens_out"], "models": {}}
    for item in usage.values():
        total_usage["calls"] += item.get("calls", 0)
        total_usage["tokens_in"] += item.get("tokens_in", 0)
        total_usage["tokens_out"] += item.get("tokens_out", 0)
        for model, row in (item.get("models") or {}).items():
            into = total_usage["models"].setdefault(model, {"calls": 0, "tokens_in": 0, "tokens_out": 0})
            for k in into:
                into[k] += row.get(k, 0)
    timing = timing_of({d.isoformat(): research[d].timing for d in days if d in research and research[d].timing}, speed=speed, research_seconds=research_seconds, walk_seconds=walk_seconds, cached_days=len(cached_days),
                       parallel_days=min(settings["parallel_days"], max(len(days) - len(cached_days), 1)), link_seconds=link_seconds, setup_seconds=setup_seconds)
    run = ProvingRun(run_id=run_id, kind="time_machine", status="stopped" if stopped else "finished", created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), seconds=round(time.perf_counter() - began, 3),
                     request={**request.model_dump(mode="json"), "cadence": cadence, "speed": speed, "intent": intent.model_dump(mode="json")}, understood=intent.describe(),
                     range=RunRange(first_day=days[0], last_day=days[-1], last_bar_day=last_bar_day), decision_days=sorted(rows, key=lambda r: r.day), equity=points, events=events, outcomes=[o.model_copy(update={"source": "time_machine"}) for o in all_outcomes],
                     stats=summary, caveats=caveats,
                     coverage={"usable": report["usable"], "last_bar_day": report["last_bar_day"], "months": report["months"], "backfill": report.get("backfill"), "timing": timing,
                               "cache": {"cached_days": cached_days, "researched_days": len(days) - len(cached_days)}, "memo": {"hits": memo.hits, "misses": memo.misses}, "notes": notes, "news_thin_days": thin_days,
                               "news_sources": {d.isoformat(): research[d].news_sources for d in days if d in research and research[d].news_sources is not None},
                               "usage": {"total": total_usage, "per_day": usage}, "desks": desks, "stamp": stamp},
                     models=models_used(), config_versions=versions(), langsmith=langsmith, speed=speed, timing=timing)
    store.save_run(run)
    emit("pv_finished", {"run_id": run_id, "status": run.status, "stats": summary.model_dump(mode="json"), "headline": summary.headline.model_dump(mode="json"), "caveats": [{"code": c, "sentence": CAVEATS[c]} for c in caveats],
                         "equity": [p.model_dump(mode="json") for p in points], "saved_id": run_id, "langsmith": langsmith.model_dump(mode="json"), "seconds": run.seconds, "timing": timing, "speed": speed})
    proving_tracing.post_in_background([(row.trace_id, [o for o in all_outcomes if o.decision_day == row.day]) for row in rows], **post_target)      # AFTER the result is out and saved: LangSmith's speed delays nothing; `wait_for_posts` is how a command waits
    return run


def timing_of(per_day: dict, *, speed: str, research_seconds: float, walk_seconds: float, cached_days: int, parallel_days: int, link_seconds: float = 0.0, setup_seconds: float = 0.0) -> dict:
    """WHERE THE SECONDS WENT (design/proving_ground_spec.md P14), from the days' own measurements: `research_seconds` and `walk_seconds` (the two phases), `per_day` ({day: {seconds, stages: {desk node: seconds}, calls: [{model, stage, start,
    seconds, tokens_in, tokens_out, ok}], attempts}}; a cached day is {seconds: 0, cached: true, researched_in}), `stages` ({desk node: {total, max}} over the researched days), `models` ({model: {calls, failed, seconds, tokens_in, tokens_out}}
    over every timed call), `setup_seconds` (what came before the research: the request read and screened, the stores warmed) and `link_seconds` (what asking LangSmith for the run's address took). Pure."""
    stages: dict[str, dict] = {}
    models: dict[str, dict] = {}
    for row in per_day.values():
        for node, seconds in (row.get("stages") or {}).items():
            into = stages.setdefault(node, {"total": 0.0, "max": 0.0})
            into["total"], into["max"] = round(into["total"] + seconds, 2), max(into["max"], seconds)
        for call in row.get("calls") or []:
            into = models.setdefault(call["model"], {"calls": 0, "failed": 0, "seconds": 0.0, "tokens_in": 0, "tokens_out": 0})
            into["calls"] += 1
            into["failed"] += 0 if call.get("ok", True) else 1
            into["seconds"] = round(into["seconds"] + call["seconds"], 2)
            into["tokens_in"] += call.get("tokens_in", 0)
            into["tokens_out"] += call.get("tokens_out", 0)
    return {"speed": speed, "research_seconds": round(research_seconds, 2), "walk_seconds": round(walk_seconds, 2), "cached_days": cached_days, "parallel_days": parallel_days, "per_day": per_day, "stages": stages, "models": models,
            "setup_seconds": round(setup_seconds, 2), "link_seconds": round(link_seconds, 2)}


def _decide_day(day: date, found: Day, intent, request: TimeMachineRequest, manager, trader, account: SimPortfolio, earlier: list[dict], book: dict, config: ProvingConfig, trader_config, clerk, info: dict) -> tuple:
    """One decision day of the walk: the cooldown, the Account Manager's decision with the account's own money, the Trader's fills at the next open and the outcomes of the day's recommendations. Returns (the day's row, the trades,
    the outcomes, the work order as a dict for the page)."""
    from prometheus.common.money import money, price
    order = WorkOrder.model_validate(found.order)
    full = order.model_copy(deep=True)
    scalars = found.plan or {}
    caught = cooldown_filter(order, day, earlier)
    beyond = beyond_count(order, scalars.get("find"))      # the first `find` after the cooldown are what a live Advisor returns; the Critic reviewed the rest too, and they are kept as hypothetical outcomes
    earlier.extend(copy.deepcopy(order.recommendations))      # a live Advisor's ledger holds what it returned, not what it passed over
    budget, each = scalars.get("budget"), scalars.get("each")
    moment = simulated_moment(day)
    with trading_clock.as_of(day):
        desk_portfolio = account.as_desk_portfolio()
        decided = manager.decide_trades(order, desk_portfolio, None, moment, "auto", scalars.get("balance"), None if budget is None else money(budget), clarify=None, said_how_much=bool(scalars.get("said_how_much")),
                                        request_text=scalars.get("request_text") or "", count=scalars.get("count"), each=None if each is None else money(each),
                                        max_price=None if scalars.get("max_price") is None else price(scalars["max_price"]),      # "under $50 a share" (entry 138): read by the same code as a live run, here the Account Manager's own check of the price
                                        min_price=None if scalars.get("min_price") is None else price(scalars["min_price"]),      # "over $20 a share" (entry 140)
                                        shares_of={name: sh.quantity(n) for name, n in (scalars.get("shares_of") or {}).items()} or None,
                                        shares_each=None if scalars.get("shares_each") is None else sh.quantity(scalars["shares_each"]))      # the shares typed with a recommendation (entry 140): each approved purchase's limit
        with trace.span("proving.fill", metadata={"as_of": day.isoformat()}):
            if decided.status in ("decided", "reviewed") and any(d.get("will_trade") in TRADING for d in decided.trade_decisions):
                decided = manager.perform_trades(decided, desk_portfolio, trader, moment, funding=None, approver=None, book=False)
    found_recs = recs.recs_of(decided, intent.model_dump(mode="json"))
    for rec in caught:
        original = {k: v for k, v in rec.items() if k != "_cooldown_of"}
        found_recs.append(recs.Rec("recommendation", original, (original.get("identity") or {}).get("record_id"), original.get("symbol"), False, None, None, "cooldown", None))
    for rec in beyond:
        found_recs.append(recs.Rec("recommendation", rec, (rec.get("identity") or {}).get("record_id"), rec.get("symbol"), False, None, None, "beyond_count", None))
    scored = []
    for rec in found_recs:
        was_caught = any((c.get("identity") or {}).get("record_id") == rec.record_id for c in caught)
        was_beyond = any((b.get("identity") or {}).get("record_id") == rec.record_id for b in beyond)
        note = (f"held back by the Advisor's cooldown: an earlier recommendation of the same strategy ({next(c['_cooldown_of'] for c in caught if (c.get('identity') or {}).get('record_id') == rec.record_id)}) still stands" if was_caught
                else f"ranked beyond the {scalars.get('find')} the request asked for: the Advisor was asked for more in case its cooldown held a better one back" if was_beyond else None)
        scored.append(track.score_rec(full if was_caught or was_beyond else decided, rec, day, book, config, trader_config, note))
        if rec.fill is not None:
            info[rec.record_id] = LotInfo(plan=record.plan_of(rec.record), key=scored[-1].key, name=rec.record.get("company"), decision_day=day)
    trades = [t for t in decided.trades if isinstance(t, dict)]
    row = track.decision_row(day, decided, found_recs, scored, None if any(o.traded for o in scored) else "nothing was bought: " + ("; ".join(d.get("reason", "")[:120] for d in decided.trade_decisions[:1]) or "no purchase was decided"))
    snapshot = json.loads(decided.model_dump_json(exclude={"data"}))
    snapshot["data"] = None
    return row, trades, scored, snapshot


def main(args) -> int:
    """The `time-machine` command: the plan, the cache hits and the cost first, then (after a yes, or `--yes`) the paid run."""
    import sys

    from prometheus.agents.account_manager.account_manager import AccountManager
    config, store = load_config(), ProvingStore()
    try:
        request = TimeMachineRequest(prompt=args.prompt, cash=args.cash, start=date.fromisoformat(args.start), end=date.fromisoformat(args.end), cadence=args.cadence, risk=args.risk, neutral=args.neutral, strict=args.strict,
                                     speed="faithful" if args.faithful else None)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    report = coverage_module.compute()
    drawn = plan(request, store, report)
    if drawn["problems"]:
        print("ERROR: the run cannot be made:\n  " + "\n  ".join(drawn["problems"]), file=sys.stderr)
        return 2
    print(f"TIME MACHINE  {request.prompt!r}  [{drawn['speed']}]\n  cash ${request.cash:,.2f}, {drawn['cadence']}, {len(drawn['decision_days'])} decision days from {drawn['decision_days'][0]} to {drawn['decision_days'][-1]}, risk {request.risk}"
          f"{', neutral counts as approved' if request.neutral and not request.strict else ''}{', strict' if request.strict else ''}\n  usable range {drawn['range']['first_day']} to {drawn['range']['last_day']}, {'every day is in the cache' if drawn['paid_days'] == 0 else str(drawn['timing']['parallel_days']) + ' days researched at once'}")
    if drawn["speed"] == "fast":
        print(f"  FAST (the default): {CAVEATS['fast_models']}")
    print("  THIS IS A PAID RUN: each decision day calls the Data Clerk's, the Advisor's and the Critic's models (the cache is read first; nothing is ever fetched).")
    if not args.yes:
        try:
            answer = input("  Go ahead? [y/N]: ")
        except EOFError:
            answer = ""
        if not answer.strip().lower().startswith("y"):
            print("  Stopped. Nothing was paid for.")
            return 1
    shown = {"pv_research": lambda d: f"  researched {d['day']}: {d['status']} ({d['recommended']} recommended, {d['approved']} approved, {d['denied']} denied; {d['seconds']} s) [{d['done']}/{d['total']}]" + (f" {d['reason']}" if d.get("reason") else ""),
             "pv_day": lambda d: None, "pv_started": lambda d: None}

    def emit(type_: str, data: dict) -> None:
        say = shown.get(type_)
        text = say(data) if say else None
        if text:
            print(text, flush=True)
    try:
        run = run_time_machine(request, store=store, emit=emit, use_cache=not args.no_cache, config=config)
    except RefusedRequest as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(track.report(run))
    timing = run.coverage["timing"]
    print(f"  saved as {run.run_id}; research {timing['research_seconds']} s ({timing['cached_days']} days from cache), walk {timing['walk_seconds']} s, total {run.seconds} s [{run.speed}]")
    for node, row in sorted(timing.get("stages", {}).items()):
        print(f"    {node:<16} {row['total']:>8.1f} s over the days, the slowest {row['max']:.1f} s")
    for model, row in sorted(timing.get("models", {}).items()):
        print(f"    {model:<40} {row['calls']:>3} calls  {row['seconds']:>8.1f} s  in {row['tokens_in']:>7,}  out {row['tokens_out']:>7,}" + (f"  ({row['failed']} failed)" if row.get("failed") else ""))
    waited = proving_tracing.wait_for_posts()      # the feedback is posted after the result: a command waits for it so that it is not lost when the process ends
    if waited["jobs"]:
        print(f"  LangSmith: {waited['rows']} feedback rows posted for {waited['jobs']} decision days ({waited['seconds']} s after the result)")
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(dumps(run.model_dump(mode="json"), indent=1), encoding="utf-8")
        print(f"(saved in {args.save})")
    return 0
