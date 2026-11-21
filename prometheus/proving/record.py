"""The shapes the Proving Ground writes and the page reads, and the readers of a real recommendation (design/proving_ground_spec.md section 8).

    plan = plan_of(recommendation)                          # a recommendation's trade plan (schema 3 on a work order, schema 2 in the older captures and the ledger) as a `PlanView`, through the price door
    verdict = verdict_of(order, record_id)                  # approved | denied | neutral | unreviewed
    who = identity_of(order, recommendation, decision_day)   # key, symbol, company, record_id, review_id, advisor, critic, verdict: the identity fields of a `RecOutcome`

    RecOutcome   one per recommendation, traded or not (8.1): the plan, the entry, the exit by the plan, the result (won | lost | open | not_filled | no_data), the returns against the benchmark and the sector, the day-by-day path
    EquityPoint  one day of the Time Machine's account (equity = cash + the positions at the close; `benchmark` and `drawdown_pct`)
    Event        a buy or an exit: the markers on the curve
    DecisionDay  what one decision day did
    Position     an open position of the simulated account on a day (the plan ladder's row)
    Exit         one position closed by the plan (`SimPortfolio.step`)
    Stats        the statistics of a run, each with its `n` and a small-sample mark (8.3: `stats.py` computes them)
    ProvingRun   one saved run (8.2): `runs/<run_id>.json`
    CAVEATS      the closed list of caveats (8.4), each ONE sentence that the page shows as written

A `_pct` is a FRACTION (0.0077 is 0.77%), as everywhere in the desk, and every `_pct` is a float: a return, a drawdown, a hit rate and a mean are measurements, worked out from Decimal values at the very end and to six places (P7).
Dollar amounts are `Money`, prices `Price` and share counts `PositiveShares` (entry 112): the doors run BEFORE pydantic's own conversion, a bool and a string are refused, and in JSON mode they are NUMBERS.

THE RECOMMENDATION IS READ AS IT IS. A recommendation's plan is `record["trade"]` (`entry_zone` [low, high], `entry_mid`, `limit_price`, `stop_price`, `target_price`, `time_stop_date`, `horizon_bars`, `order_style`: the fields `tickets.py`
reads), its review is the Critic's record on `order.reviews` (`approved`, `score`, `grade`, `denial_reasons`), its id is `identity.record_id`. A price crosses the price door once, and what is not a price above zero (a capital-return plan has no target) is None.
THE VERDICT: a review that approved is `approved`, one that did not is `denied` (entry 103: the Critic's verdict is approved or denied, nothing between), a company the Advisor scored neutral and the Account Manager decided on
(`origin: "neutral"`) is `neutral`, anything the Critic did not review is `unreviewed`.
Additions to the spec's lists, each for a reason: `PlanView.entry_mid` and `limit_price` (the Trader's own order rules need them: `tickets.limit_range`, `tickets.default_order`), `RecOutcome.reject_code` (the Trader's reason code of a
`not_filled`), `RecOutcome.split_adjusted` (the caveat of that name is decided from it) and `Stats.portfolio.start_equity` and `end_equity`.
"""
from datetime import date as Date
from decimal import Decimal
from typing import Annotated, Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from prometheus.common.money import Jsonable, Money, NonNegativeMoney, Price, price as price_door
from prometheus.common.shares import PositiveShares
from prometheus.common.work_order import iso_date, review_of

Measure = float      # a return, a drawdown, a rate, a mean or a timing: a measurement, never added to money (P7). Every `_pct` below is a Measure

Verdict = Literal["approved", "denied", "neutral", "unreviewed"]
Result = Literal["won", "lost", "open", "not_filled", "no_data"]
ExitReason = Literal["stop", "target", "time_stop"]
NotTradedReason = Literal["denied", "neutral_off", "beyond_count", "no_cash", "not_filled", "failed", "cooldown"]      # `cooldown`: the Advisor's own cooldown held it back (the engine applies it, spec 7.3)
Speed = Literal["fast", "faithful"]
OutcomeSource = Literal["run", "fixture", "time_machine"]      # where the outcome's recommendation came from: a saved live run, a recorded capture, or a Time Machine decision day
RunKind = Literal["time_machine", "track", "live"]
RunStatus = Literal["finished", "stopped", "failed"]
DayStatus = Literal["traded", "no_trade", "failed", "cached"]
Tone = Literal["pos", "neg", "neutral"]
CaveatCode = Literal["fundamentals_not_point_in_time", "symbol_directory_is_todays", "price_returns_no_dividends", "fills_at_next_open_no_slippage", "daily_bars_stop_before_target", "news_coverage_thin",
                     "small_sample", "open_positions_marked", "split_adjusted", "model_knowledge_cutoff", "fast_models"]

# 8.4: the closed list of caveats. Code decides which apply; each has ONE sentence, and the page shows it as written (P3: never hidden).
CAVEATS: dict[str, str] = {
    "fundamentals_not_point_in_time": "Company fundamentals (earnings, margins, ratios) are today's figures from Finnhub, not the figures known on each past decision day.",
    "symbol_directory_is_todays": "Companies are found with today's symbol directory, so a company that was delisted or renamed since a decision day cannot be found.",
    "price_returns_no_dividends": "Returns are price returns with no dividends, for the stocks and for the benchmark alike.",
    "fills_at_next_open_no_slippage": "Every purchase fills at the next regular open of the stored daily bars, with no slippage and no commission.",
    "daily_bars_stop_before_target": "Exits are judged on daily bars: a day that touches both the stop and the target counts as the stop, and on the day of the entry only the stop can fire.",
    "news_coverage_thin": "On some decision days few stored articles were found for the companies, so the research of those days saw less than it would have live.",
    "small_sample": "There are fewer outcomes than the minimum sample, so every rate and average is a first look, not a verdict.",
    "open_positions_marked": "A position that no rule has closed is open: it is marked at the last stored close and is never counted as a win.",
    "split_adjusted": "A position that crossed a stock split was adjusted on the split's date from the stored split table.",
    "model_knowledge_cutoff": "The models know the world only up to their own training cutoff, which can be later than a past decision day: they may remember what happened next.",
    "fast_models": "The Advisor's model ran with its reasoning switched off and the Critic's at reduced reasoning effort to make this test fast; a faithful run uses the live settings and takes longer.",
}
assert set(get_args(CaveatCode)) == set(CAVEATS), "the Literal of caveat codes and the closed list of sentences must say the same"


def caveat_sentences(codes) -> list[str]:
    """The one sentence of each caveat code, in the order given. Raises ValueError for a code that is not in the closed list."""
    unknown = [c for c in codes if c not in CAVEATS]
    if unknown:
        raise ValueError(f"not a caveat of the closed list: {unknown[0]!r}")
    return [CAVEATS[c] for c in codes]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- one recommendation, followed

class PlanView(_Model):
    """The Advisor's trade plan as the Proving Ground reads it: prices (through the door) and dates, and NO size."""
    entry_low: Price | None = None
    entry_high: Price | None = None
    entry_mid: Price | None = None          # the Advisor's reference price (its last close): what the Trader's ticket measures the limit range from when the Critic gave no deal price
    limit_price: Price | None = None
    stop: Price | None = None
    target: Price | None = None
    time_stop_date: Date | None = None
    horizon_bars: int | None = None
    order_style: str | None = None


class AdvisorCell(_Model):
    score: int | None = None
    stance: str | None = None
    family: str | None = None
    rank: int | None = None


class CriticCell(_Model):
    approved: bool
    score: int | None = None
    grade: str | None = None
    denial_reasons: list[str] = Field(default_factory=list)
    # What the review found, by code, and the version of the Critic that made it (the report card of the Critic's rules, `report_card.py`, scores each code against the outcomes). None: the run was saved before they were kept
    # (the report card then reads them from the review in the research cache); [] is a review that found none.
    config_version: str | None = None
    blockers: list[str] | None = None
    warnings: list[str] | None = None
    human: list[str] | None = None            # the codes of the findings that asked for a person (`requires_human_reasons`): a denial since entry 103, and a warning only since Critic version 2 for fresh_negative_news


class EntryView(_Model):
    date: Date
    price: Price
    shares: PositiveShares          # a hypothetical outcome is ONE share-equivalent
    amount: Money


class ExitView(_Model):
    date: Date
    price: Price
    reason: ExitReason
    sessions_held: int


class Horizon(_Model):
    """The position's return at `sessions` sessions after the entry bar (the bar of the entry is session 0), with the benchmark's over the same dates: by the plan, so a position that closed before that session keeps its realized return and
    the benchmark is read to the day it closed; None for both while the position is still open and has not been held that long."""
    sessions: int
    return_pct: Measure | None = None
    benchmark_return_pct: Measure | None = None


class Path(_Model):
    """One day of a position's path: the close, the return since the entry price and the benchmark's return since the open of the entry day (the one `simulator.follow` uses). (The spec calls the model `Path`; a list of them is `RecOutcome.path`.)"""
    date: Date
    close: Price
    return_pct: Measure
    benchmark_return_pct: Measure | None = None


class RecOutcome(_Model):
    key: str = Field(min_length=1)                           # run-unique
    decision_day: Date
    symbol: str
    company: str | None = None
    record_id: str | None = None
    review_id: str | None = None
    advisor: AdvisorCell | None = None
    critic: CriticCell | None = None
    trust_score: Measure | None = None
    verdict: Verdict = "unreviewed"
    traded: bool
    hypothetical: bool                                       # P8: not traded; one share-equivalent bought at the next open and followed by its plan; never in the equity curve
    not_traded_reason: NotTradedReason | None = None
    source: OutcomeSource | None = None                       # where the recommendation came from (the Track Record sets "run" or "fixture", a Time Machine day "time_machine"); None in a run saved before the field existed
    reject_code: str | None = None                           # a `not_filled`: the Trader's own code (limit_not_reached, price_outside_limits, price_unavailable)
    plan: PlanView
    entry: EntryView | None = None
    exit: ExitView | None = None
    result: Result
    return_pct: Measure | None = None                          # closed: against the entry price; open: against the last close
    benchmark_return_pct: Measure | None = None
    excess_return_pct: Measure | None = None
    sector_return_pct: Measure | None = None
    gain: Money | None = None                                # realized, or unrealized at the last close; None when hypothetical
    horizons: list[Horizon] = Field(default_factory=list)
    max_favorable_pct: Measure | None = None                   # the best close since the entry, against the entry price
    max_adverse_pct: Measure | None = None                     # the worst
    path: list[Path] = Field(default_factory=list)           # the entry day to the exit or the last bar
    split_adjusted: bool = False
    notes: list[str] = Field(default_factory=list)           # a split adjusted, why no data, a break-even

    @model_validator(mode="after")
    def _the_parts_agree(self):
        if self.traded == self.hypothetical:
            raise ValueError("an outcome is traded or hypothetical, never both and never neither")
        if self.hypothetical and self.gain is not None:
            raise ValueError("a hypothetical outcome has no gain in money: nothing was bought")
        if self.result in ("won", "lost") and (self.entry is None or self.exit is None):
            raise ValueError("a closed outcome has its entry and its exit")
        if self.result == "open" and (self.entry is None or self.exit is not None):
            raise ValueError("an open outcome has an entry and no exit")
        if self.result in ("not_filled", "no_data") and (self.entry is not None or self.exit is not None):
            raise ValueError("an outcome that was not filled or has no data has no entry and no exit")
        return self


# ---------------------------------------------------------------- the account of a run

class Position(_Model):
    """An open position of the simulated account, in the share basis of TODAY (after any split): the plan ladder's row. `price` is the last mark, `stale` when that mark is not from the day asked about."""
    symbol: str
    name: str | None = None
    shares: PositiveShares
    cost_basis: Price
    entry_date: Date
    record_id: str | None = None
    key: str | None = None
    stop: Price | None = None
    target: Price | None = None
    time_stop_date: Date | None = None
    price: Price | None = None
    value: Money | None = None
    gain: Money | None = None
    stale: bool = False


class Exit(_Model):
    """One position closed by its plan (`SimPortfolio.step`): the sale at the rule's price, in the share basis of the day it closed. `amount` is what came into the cash and `gain` is the amount less the cost of the lot, each to the cent
    (`trade_amount`); a lot that crossed a split is worked in the share basis of its ENTRY, so shares x price can differ from `amount` by a rounding of the price (4 places)."""
    date: Date
    symbol: str
    shares: PositiveShares
    price: Price
    amount: Money
    cost: Money
    gain: Money
    reason: ExitReason
    sessions_held: int
    entry_date: Date
    record_id: str | None = None
    key: str | None = None


class EquityPoint(_Model):
    date: Date
    equity: Money                                            # cash + the positions at the close
    cash: Money
    invested: Money
    benchmark: Money | None = None                           # the same starting cash put into the benchmark (the engine's curve)
    drawdown_pct: Measure = Field(default_factory=Measure)                                # a fraction at or below zero: equity against its peak (the starting cash counts)
    stale: list[str] = Field(default_factory=list)           # the held symbols with no bar this day: marked at their previous close


class Event(_Model):
    date: Date
    kind: Literal["buy", "exit"]
    symbol: str
    shares: PositiveShares
    price: Price
    amount: Money
    reason: str | None = None
    key: str | None = None


class DecisionDay(_Model):
    day: Date
    status: DayStatus
    recommended: int = 0
    approved: int = 0
    denied: int = 0
    traded: int = 0
    spent: NonNegativeMoney = Decimal("0.00")
    reason: str | None = None
    trace_id: str | None = None


class RunRange(_Model):
    first_day: Date
    last_day: Date
    last_bar_day: Date | None = None


# ---------------------------------------------------------------- the statistics (8.3)

class PortfolioStats(_Model):
    start: Date | None = None                                # the first and the last day of the curve
    end: Date | None = None
    start_equity: Money                                      # the starting cash
    end_equity: Money
    return_pct: Measure
    benchmark_return_pct: Measure | None = None
    excess_return_pct: Measure | None = None
    max_drawdown_pct: Measure                                # a fraction at or below zero
    drawdown_from: Date | None = None                        # the peak the worst drawdown fell from, and the day of its bottom
    drawdown_to: Date | None = None
    best_day_pct: Measure | None = None
    worst_day_pct: Measure | None = None
    days: int
    invested_avg_pct: Measure | None = None


class Exits(_Model):
    stop: int = 0
    target: int = 0
    time_stop: int = 0
    open: int = 0


class OutcomeStats(_Model):
    n: int                                                   # every outcome counted: n = closed + open + not_filled + no_data
    closed: int
    open: int
    won: int
    lost: int
    not_filled: int = 0
    no_data: int = 0
    win_rate: Measure | None = None                            # won / closed: an open position is never a win; None with nothing closed
    avg_win_pct: Measure | None = None
    avg_loss_pct: Measure | None = None
    profit_factor: Measure | None = None                       # the sum of the winning returns over the sum of the losing ones; None without a loss
    avg_sessions_held: Measure | None = None                   # over the closed
    exits: Exits
    small_sample: bool                                       # fewer closed outcomes than the minimum: no colour of success or failure (P9)


class GroupStats(_Model):
    """One group of outcomes (a verdict, a score bucket, a family): `n` is the outcomes that have a return and an excess over the benchmark (what the means rest on), `closed` those of them that closed (what the win rate rests on)."""
    n: int
    closed: int
    mean_return_pct: Measure | None = None
    mean_excess_pct: Measure | None = None
    win_rate: Measure | None = None
    small_sample: bool


class VerdictStats(GroupStats):
    verdict: Verdict


class ScoreStats(GroupStats):
    low: int                                                 # the Advisor's score bucket [low, high)
    high: int


class FamilyStats(GroupStats):
    family: str


class SymbolStats(_Model):
    symbol: str
    n: int                                                   # the traded outcomes of the symbol
    gain: Money | None = None
    return_pct: Measure | None = None                          # the gain over what was paid (money-weighted)


class CriticValue(_Model):
    denied_n: int
    denied_mean_excess_pct: Measure | None = None
    approved_n: int
    approved_mean_excess_pct: Measure | None = None
    sentence: str                                            # code's own, from a closed set of templates (`stats.CRITIC_SENTENCES`)


class Headline(_Model):
    tone: Tone
    sentence: str                                            # code's own, from a closed set of templates (`stats.HEADLINE_LEADS` and `HEADLINE_TAILS`)


class Stats(_Model):
    portfolio: PortfolioStats | None = None                  # None without an equity curve (the Track Record has none of its own)
    outcomes: OutcomeStats
    by_verdict: list[VerdictStats]
    by_score: list[ScoreStats]
    by_family: list[FamilyStats]
    by_symbol: list[SymbolStats]
    critic_value: CriticValue
    headline: Headline


# ---------------------------------------------------------------- one saved run (8.2)

class LangSmith(_Model):
    trace_id: str | None = None
    url: str | None = None


class ProvingRun(_Model):
    run_id: str = Field(min_length=1)
    kind: RunKind
    status: RunStatus
    created_utc: str
    seconds: Measure = Field(default_factory=Measure)                                     # a timing
    request: Annotated[dict[str, Any], Jsonable] | None = None      # the TimeMachineRequest as the engine dumped it
    understood: str = ""                                     # the Account Manager's own words for what it read
    range: RunRange | None = None
    decision_days: list[DecisionDay] = Field(default_factory=list)
    equity: list[EquityPoint] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    outcomes: list[RecOutcome] = Field(default_factory=list)
    stats: Stats
    caveats: list[CaveatCode] = Field(default_factory=list)
    coverage: Annotated[dict[str, Any], Jsonable] = Field(default_factory=dict)
    models: Annotated[dict[str, Any], Jsonable] = Field(default_factory=dict)
    config_versions: Annotated[dict[str, Any], Jsonable] = Field(default_factory=dict)
    langsmith: LangSmith | None = None
    speed: Speed = "faithful"                                # how the models were asked (a run saved before the modes existed was made with the live settings: faithful)
    timing: Annotated[dict[str, Any], Jsonable] = Field(default_factory=dict)      # where the seconds went: {research_seconds, walk_seconds, cached_days, parallel_days, speed, per_day: {day: {seconds, stages, calls}}, models: {model: {calls, seconds, tokens_in, tokens_out}}, posts}


# ---------------------------------------------------------------- reading a real recommendation

def _price(measured) -> Decimal | None:
    """A plan price through THE price door, once (four places, half up); None when it is not a number above zero (a capital-return plan has no target)."""
    try:
        return price_door(measured)
    except ValueError:
        return None


def _whole(value) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def plan_of(recommendation) -> PlanView:
    """The trade plan of a recommendation record (schema 3 on a work order; schema 2 in the older captures and the ledger: the plan is the same), as a `PlanView`. A record with no plan (a neutral company the Advisor found no
    strategy for) gives a plan with nothing in it: nothing invents a stop or a target. A NEUTRAL company's record carries an `exit_plan` since Advisor config version 2 (a stop and a time stop, no target, nothing about the entry:
    `trade_plan.neutral_exit_plan`): then the plan is that stop and that time stop and nothing else, so a neutral position is followed by them (`simulator.follow`)."""
    plan = recommendation.get("trade") if isinstance(recommendation, dict) else None
    plan = plan if isinstance(plan, dict) else {}
    exit_plan = recommendation.get("exit_plan") if isinstance(recommendation, dict) else None
    if isinstance(exit_plan, dict) and not any(plan.get(k) is not None for k in ("stop_price", "target_price", "time_stop_date")):
        return PlanView(stop=_price(exit_plan.get("stop_price")), time_stop_date=iso_date(exit_plan.get("time_stop_date")))
    zone = plan.get("entry_zone")
    low, high = (_price(zone[0]), _price(zone[1])) if isinstance(zone, (list, tuple)) and len(zone) == 2 else (None, None)
    style = plan.get("order_style")
    return PlanView(entry_low=low, entry_high=high, entry_mid=_price(plan.get("entry_mid")), limit_price=_price(plan.get("limit_price")), stop=_price(plan.get("stop_price")), target=_price(plan.get("target_price")),
                    time_stop_date=iso_date(plan.get("time_stop_date")), horizon_bars=_whole(plan.get("horizon_bars")), order_style=style if isinstance(style, str) else None)


def verdict_of(order, record_id: str) -> str:
    """What the Critic's side of the desk said about a recommendation: `approved` or `denied` (its review: `approved` true or false; a review that asks for a person is a denial, entry 103), `neutral` (no review, and the Account
    Manager decided on a company the Advisor scored neutral: `origin: "neutral"`), else `unreviewed`."""
    review = review_of(order, record_id)
    if review is not None:
        return "approved" if review.get("approved") is True else "denied"
    if any(isinstance(d, dict) and d.get("record_id") == record_id and d.get("origin") == "neutral" for d in order.trade_decisions):
        return "neutral"
    return "unreviewed"


def advisor_cell(recommendation) -> AdvisorCell:
    """The Advisor's score, stance, family and rank of a recommendation record (or of a candidate, which has `rank` at the top)."""
    rec = recommendation if isinstance(recommendation, dict) else {}
    plan = rec.get("trade") if isinstance(rec.get("trade"), dict) else {}
    ranking = rec.get("ranking") if isinstance(rec.get("ranking"), dict) else {}
    score, rank = rec.get("score"), ranking.get("rank", rec.get("rank"))
    return AdvisorCell(score=score if isinstance(score, int) and not isinstance(score, bool) else None, stance=rec.get("stance") if isinstance(rec.get("stance"), str) else None,
                       family=plan.get("family") if isinstance(plan.get("family"), str) else None, rank=rank if isinstance(rank, int) and not isinstance(rank, bool) else None)


def critic_cell(review) -> CriticCell | None:
    """The Critic's verdict of a review record, or None without one."""
    if not isinstance(review, dict) or not isinstance(review.get("approved"), bool):
        return None
    score = review.get("score")
    reasons = review.get("denial_reasons")
    codes = lambda key: [f["code"] for f in review.get(key) or [] if isinstance(f, dict) and isinstance(f.get("code"), str)] if isinstance(review.get(key), list) else None
    human = review.get("requires_human_reasons")
    version = (review.get("audit") or {}).get("config_version") if isinstance(review.get("audit"), dict) else None
    return CriticCell(approved=review["approved"], score=score if isinstance(score, int) and not isinstance(score, bool) else None, grade=review.get("grade") if isinstance(review.get("grade"), str) else None,
                      denial_reasons=[r for r in reasons if isinstance(r, str)] if isinstance(reasons, list) else [], config_version=version if isinstance(version, str) else None, blockers=codes("blockers"),
                      warnings=codes("warnings"), human=[r for r in human if isinstance(r, str)] if isinstance(human, list) else None)


def identity_of(order, recommendation, decision_day: Date) -> dict:
    """The identity fields of a `RecOutcome` for one recommendation of a work order: key (the decision day and the record id: run-unique), decision_day, symbol, company, record_id, review_id, advisor, critic and verdict. What the
    recommendation was and what the desk said about it; nothing about how it did."""
    rec = recommendation if isinstance(recommendation, dict) else {}
    record_id = (rec.get("identity") or {}).get("record_id") or rec.get("record_id")
    review = review_of(order, record_id) if isinstance(record_id, str) else None
    symbol = rec.get("symbol")
    return {"key": f"{decision_day.isoformat()}:{record_id or symbol}", "decision_day": decision_day, "symbol": symbol, "company": rec.get("company") if isinstance(rec.get("company"), str) else None,
            "record_id": record_id if isinstance(record_id, str) else None, "review_id": review.get("review_id") if review else None, "advisor": advisor_cell(rec), "critic": critic_cell(review),
            "verdict": verdict_of(order, record_id) if isinstance(record_id, str) else "unreviewed"}
