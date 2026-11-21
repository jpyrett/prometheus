"""Value: is the price a good deal? (Critic v1 spec section 8.5, step 8). No network, no model.

    result = assess(rec, fetched, deal_price, config.value, unknown_credit)

The Critic's own view of the price, from the numbers it just had the Data Clerk calculate (spec P2): where the price sits against its SIX-MONTH average (126 sessions), whether the valuation is
sane, and whether the stock is collapsing (a collapse is not a discount). A bad deal is a BLOCKER, so the Account Manager records the denial
on the work order; a great deal earns a bonus on the score, in proportion to the discount to the six-month average.

    premium  = deal_price / sma_126 - 1          discount = -premium
    label    falling | unknown | overpriced | rich | great | good | fair      (the first that fits)

THE REWARD OF `reward_to_risk` IS THE MARKET'S, NOT THE PLAN'S (design/advisor_exit_plan_spec.md rule E7): it is the distance from the deal price up to the 126-session high over the plan's stop distance, so it reads no `target_price`. A TRAILING plan (the
TREND family since Advisor config version 3: `trailing` true, no target, a stop that rises with the highest close) therefore needs no special case: the measure is the same number it was for any plan, a missing target is never read as a zero reward and
never as an unknown (the unknown credit is for a missing 126-session high or stop distance only), and `stop_distance` is still the plan's starting stop distance.

A company with fewer than 126 stored bars (a recent IPO) has no six-month average. That is a plain NOTE on the review (`value_not_assessable`, a warning), never a denial: it does not ask for a person (design choices entry 103,
corrected: Jacob, 2026-11-15, "it eliminates IPOs from being traded and is a bad rule"), the measures that need the average count the unknown credit as they always did, and the other value rules still apply.
"""
from dataclasses import dataclass, field
from datetime import date, timedelta

from prometheus.agents.advisor.record import Recommendation
from prometheus.agents.critic.common import blocker, clip, metric_value, pct, warning
from prometheus.agents.critic.config import Bonus, Value
from prometheus.agents.critic.record import Finding, Measure, ValueDetail


@dataclass
class ValueResult:
    findings: list[Finding] = field(default_factory=list)
    score: float = 0.0
    measures: list[Measure] = field(default_factory=list)
    detail: ValueDetail = field(default_factory=ValueDetail)
    applicable: bool = True
    discount: float | None = None


def label_of(premium: float | None, falling: bool, config: Value) -> str:
    """The deal, in one word, the first that fits: falling (a collapse), unknown (no six-month average), overpriced, rich, great, good, fair."""
    if falling:
        return "falling"
    if premium is None:
        return "unknown"
    discount = -premium
    if premium > config.max_premium:
        return "overpriced"
    if premium > config.rich_premium:
        return "rich"
    if discount >= config.great_discount:
        return "great"
    if discount >= config.good_discount:
        return "good"
    return "fair"


def bonus_points(discount: float | None, config: Bonus) -> float:
    """`max_points x clip((discount - zero_at) / (full_at - zero_at))`: nothing for a price within `zero_at` of its six-month average, the full bonus from `full_at` under it."""
    if discount is None:
        return 0.0
    return round(config.max_points * clip((discount - config.zero_at) / (config.full_at - config.zero_at)), 2)


def assess(rec: Recommendation, fetched: dict | None, deal_price: float | None, config: Value, unknown_credit: float) -> ValueResult:
    """The value findings of one record, its score (0 to 100), the label and the detail."""
    out = ValueResult()
    avg, high, low = (metric_value(fetched, n) for n in ("sma_126", "high_126d", "low_126d"))
    ey, eps, ps = (metric_value(fetched, n) for n in ("earnings_yield", "eps", "ps_ratio"))
    ret_21, drawdown = metric_value(fetched, "ret_21d"), metric_value(fetched, "max_drawdown_63d")
    stop_distance = rec.trade.stop_distance
    premium = round(deal_price / avg - 1, 6) if deal_price and avg else None      # six places: 130 against 100 is 30%, not 30.000000000000004%
    discount = None if premium is None else -premium
    reward = (high - deal_price) / stop_distance if high is not None and deal_price and stop_distance and stop_distance > 0 else None
    falling = (ret_21 is not None and ret_21 <= config.knife_ret_21d) or (drawdown is not None and drawdown <= config.knife_drawdown_63d)
    label = label_of(premium, falling, config)
    out.discount = discount

    if premium is not None and premium > config.max_premium:
        out.findings.append(blocker("value", "overpriced", f"The price {deal_price:.2f} is {pct(premium)} above its six-month average of {avg:.2f} (the limit is {pct(config.max_premium, 0)}).", value=premium, threshold=config.max_premium))
    if (ey is not None and ey < config.min_earnings_yield) or (eps is not None and eps <= 0 and ps is not None and ps > config.max_ps_when_unprofitable):
        why = (f"the earnings yield is {pct(ey, 2)} (a P/E of {1 / ey:.0f}: the limit is {1 / config.min_earnings_yield:.0f})" if ey is not None and ey < config.min_earnings_yield
               else f"it has no profit (EPS {eps:g}) and trades at {ps:.1f} times sales (the limit is {config.max_ps_when_unprofitable:g})")
        out.findings.append(blocker("value", "valuation_extreme", f"The valuation is extreme: {why}.", value=ey if ey is not None else ps))
    if falling:
        moves = []
        if ret_21 is not None and ret_21 <= config.knife_ret_21d:
            moves.append(f"it fell {abs(ret_21):.0%} in 21 sessions")
        if drawdown is not None and drawdown <= config.knife_drawdown_63d:
            moves.append(f"it is {abs(drawdown):.0%} below its 63-session peak")
        out.findings.append(blocker("value", "falling_knife", f"{' and '.join(moves).capitalize()}: a collapse is not a discount.", value=[ret_21, drawdown]))
    if avg is None:
        out.findings.append(warning("value", "value_not_assessable", f"The price cannot be judged against a six-month average: {(fetched or {}).get('sma_126', {}).get('note') or 'no value'}."))
    elif label == "rich":
        out.findings.append(warning("value", "rich_price", f"The price is {pct(premium)} above its six-month average ({avg:.2f}).", value=premium, threshold=config.rich_premium))

    # the score: four measures, each from 0 (zero_at) to 1 (full_at); one that could not be made counts the unknown credit
    values = {"discount_to_avg_6m": discount, "reward_to_risk": reward, "earnings_yield": ey}
    total = 0.0
    for m in config.measures:
        x = values[m.name]
        points = unknown_credit if x is None else clip((x - m.zero_at) / (m.full_at - m.zero_at))
        total += m.weight * points
        out.measures.append(Measure(name=m.name, x=None if x is None else round(x, 6), value=round(points, 4), weight=m.weight, available=x is not None,
                                    note=None if x is not None else "could not be measured: counted as unknown"))
    out.score = round(total, 2)
    out.detail = ValueDetail(deal=label, avg_6m=avg, high_6m=high, low_6m=low, premium=premium, discount=discount, reward_to_risk=reward, earnings_yield=ey)
    return out


# ---------------------------------------------------------------- two warnings from history (spec 8.6): neither changes a score

def history_warnings(rec: Recommendation, deal_price: float | None, history, today: date, config: Value) -> list[Finding]:
    """`price_above_last_recommendation`: the same symbol was recommended in the last `last_recommendation_days` at an entry well under today's price (the Advisor's ledger, read only).
    `history` is the Advisor's `Ledger` (or None). (`adding_above_cost`, a warning that read the cost basis of the user's holding, is gone: the Critic never touches the holdings, entry 105.)"""
    out: list[Finding] = []
    if deal_price is None:
        return out
    if history is not None:
        earlier = [r for r in history.recent(rec.symbol, since=today - timedelta(days=config.last_recommendation_days))
                   if r.get("identity", {}).get("record_id") != rec.identity.record_id and (r.get("trade") or {}).get("entry_mid")]
        if earlier:
            before = earlier[-1]["trade"]["entry_mid"]
            if deal_price > before * (1 + config.last_recommendation_rise):
                out.append(warning("value", "price_above_last_recommendation", f"{rec.symbol} was recommended at an entry of {before:.2f} on {earlier[-1]['identity']['as_of_utc'][:10]}; the price is now "
                                   f"{pct(deal_price / before - 1)} higher.", value=deal_price, threshold=before))
    return out
