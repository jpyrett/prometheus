"""The trade plan: where to enter, where the idea is wrong, and when to stop waiting (Advisor v1 spec section 10, design choices entries 51 and 80).

    plan = build(family, metrics, quote, today, config)      # a TradePlan

All arithmetic, no model. The plan is the part of a recommendation a Critic can check with a calculator, so every number comes with its basis:

    entry_zone      the last price +- 0.5 ATR; the last price is the quote when the order has one, else the last stored close (`entry_basis` says which)
    stop_distance   2 ATR for reversion and event, 3 ATR for trend and capital_return (or the distance to the 50-day mean when that is nearer and below the price); at least the typical
                    earnings move when a report falls inside the horizon (`gap_risk`; the earnings dates are not available in this build, so it never fires)
    stop_price      the entry mid minus the stop distance (long only: a bearish call is not a recommendation, not a short)
    target_price    the entry mid x (1 + 1.5 x the family's edge at its horizon); none for capital_return (it is bought for the payout, not for a target)
    time_stop_date  `horizon_bars` weekdays after today (the calendar of the future has no holidays in it: the note says so)
    order_style     limit at the entry mid for reversion, for capital_return, for an event that has moved less than one ATR; a marketable limit
                    at the last price + 0.25 ATR for trend while it is above its 200-day mean; a limit otherwise
    invalidation    the rules that say "this idea is wrong or finished": the price under the stop, the time stop, a bearish event, and the family's own condition

THERE IS NO SIZE (entry 80). The Advisor says which companies, at what prices and until when; HOW MUCH to buy is the Account Manager's (`account_manager/sizing.py`: the user's money). A plan is for a
PURCHASE: there is no plan for a sale (the Advisor recommends none, entry 105). The plan therefore holds no number of shares, no risk budget and no share of the account, and `build` is given neither the portfolio nor the equity: the Advisor does not use the
user's money. The stop DISTANCE stays in the plan because it is a price fact of the idea (the Account Manager's loss limit divides by it). The regime (stress or unknown) is still context for the
families and the record's identity; it used to halve the risk budget, and with no budget it has no part in the plan, so `build` is not given it.

ONE READING THE SPEC LEAVES OPEN, flagged in design choices entry 51: "the family's edge at its horizon": the 20-day edge is taken as it is for a horizon of 20 bars or less (the
reversion edge is realised inside its 10 bars, spec 8.5) and scaled up in proportion for a longer one (trend 63 bars = 3.15 x), so a long-horizon target is not smaller than its own stop.
(The other reading of entry 51, the share count formula, left with the size.)
"""
from datetime import date, timedelta

from prometheus.agents.advisor.config import AdvisorConfig
from prometheus.agents.advisor.events import BEARISH
from prometheus.agents.advisor.gates import value
from prometheus.agents.advisor.record import ExitPlan, FamilyResult, TradePlan
from prometheus.common.market_calendar import next_trading_day

FAMILY_INVALIDATION = {"trend": "below_sma_200", "reversion": "rsi_14_above_70"}   # capital_return and conviction have none of their own: the stop, the time and a bearish event end them


def add_weekdays(day: date, n: int) -> date:
    """The date `n` weekdays after `day` (holidays are not known for a future date)."""
    while n > 0:
        day += timedelta(days=1)
        n -= day.weekday() < 5
    return day


def add_trading_days(day: date, n: int) -> date:
    """The date `n` TRADING days after `day`, by the market calendar (weekends and the exchange's holidays skipped: `market_calendar.next_trading_day`)."""
    for _ in range(n):
        day = next_trading_day(day)
    return day


def neutral_exit_plan(metrics: dict | None, today: date, config: AdvisorConfig) -> ExitPlan:
    """The exit plan of a NEUTRAL company (Advisor v2, design/critic_v2_evidence_spec.md section 7): the stop is `neutral.stop_atr_multiple` (2.0) times the 14-day ATR under the last close, and the time stop is `neutral.horizon_bars` (20)
    trading days after `today`, the decision day. NO target (nothing in the record says where a neutral company should go) and nothing else: it is an EXIT plan, not an entry plan, so the Trader's tickets of a neutral purchase (market, at the
    quote) are untouched. With no ATR (fewer than 20 stored bars) or a stop that would not be above zero there is no stop and the time stop alone stands, and `basis` says so. Pure arithmetic, no model, and no size, no portfolio."""
    rule = config.neutral
    last, atr = value(metrics, "last_close"), value(metrics, "atr_14")
    stop = round(last - rule.stop_atr_multiple * atr, 4) if last is not None and atr is not None and last - rule.stop_atr_multiple * atr > 0 else None
    when = add_trading_days(today, rule.horizon_bars).isoformat()
    if stop is not None:
        basis = f"stop {rule.stop_atr_multiple:g} x atr_14 ({atr:.2f}) under the last close {last:.2f}; time stop {rule.horizon_bars} trading days after {today.isoformat()}; no target"
    else:
        basis = (f"no stop: {'atr_14 is missing (needs 20 bars)' if atr is None or last is None else 'the stop would not be above zero'}; the time stop alone stands, {rule.horizon_bars} trading days after {today.isoformat()}; no target")
    return ExitPlan(stop_price=stop, time_stop_date=when, basis=basis)


def price_of(symbol: str, metrics: dict | None, quote: dict | None) -> tuple[float | None, str]:
    """(the price, where it came from): the order's quote, else the last stored close."""
    if quote and isinstance(quote.get("price"), (int, float)) and quote["price"] > 0:
        return float(quote["price"]), "quote"
    last = value(metrics, "last_close")
    return (float(last), "last_close") if last else (None, "none")


def build(family: FamilyResult, metrics: dict | None, quote: dict | None, today: date, config: AdvisorConfig, symbol: str = "", reaction_atr: float | None = None) -> TradePlan:
    """The plan for one symbol on `family`, the strategy it is bought on. No plan has a size (module text)."""
    tc = config.trade
    last, basis = price_of(symbol, metrics, quote)
    atr, sma50 = value(metrics, "atr_14"), value(metrics, "sma_50")
    plan = TradePlan(family=family.family, entry_basis=basis, horizon_bars=family.horizon_bars)
    if last is None or atr is None:
        plan.notes.append("no plan prices: " + ("there is no price" if last is None else "atr_14 is missing (needs 20 bars)"))
        return plan
    mid = last
    plan.entry_mid, plan.entry_zone = round(mid, 4), [round(last - tc.entry_atr * atr, 4), round(last + tc.entry_atr * atr, 4)]
    distance = tc.stop_atr[family.family] * atr
    if family.family in ("trend", "capital_return") and sma50 is not None and sma50 < mid and mid - sma50 < distance:
        distance = mid - sma50
        plan.notes.append(f"the stop is the 50-day mean {sma50:.2f}, nearer than {tc.stop_atr[family.family]:g} ATR")
    days, move = value(metrics, "days_to_next_earnings"), value(metrics, "earnings_move_abs")
    if days is not None and move is not None and days <= family.horizon_bars:
        plan.gap_risk, distance = True, max(distance, move * last)
    plan.stop_distance, plan.stop_price = round(distance, 4), round(mid - distance, 4)
    plan.notes.append(f"entry zone = last +- {tc.entry_atr:g} x atr_14; stop = {tc.stop_atr[family.family]:g} x atr_14 (atr_14 = {atr:.2f})" + ("" if distance == tc.stop_atr[family.family] * atr else "; the stop distance was changed, see the notes"))
    edge = (family.edge_20d or 0.0) * max(1.0, family.horizon_bars / 20)
    plan.target_price = None if family.family == "capital_return" else round(mid * (1 + tc.target_edge_multiple * edge), 4)
    plan.time_stop_date = add_weekdays(today, family.horizon_bars).isoformat()
    plan.time_stop_note = "weekdays only: holidays of the coming weeks are not known"
    style = "limit"
    if family.family == "trend" and value(metrics, "above_sma_200") is True:
        style = "marketable_limit"
    elif family.family == "event" and (reaction_atr is None or reaction_atr >= tc.event_limit_reaction_atr):
        plan.notes.append("order style: a limit at the entry mid (the move since the event is one ATR or more, or unknown)")
    plan.order_style = style
    plan.limit_price = round(last + tc.trend_limit_atr * atr, 4) if style == "marketable_limit" else round(mid, 4)
    plan.invalidation = [{"kind": "price_below", "value": plan.stop_price}, {"kind": "days_elapsed", "value": family.horizon_bars}, {"kind": "event_class", "value": list(BEARISH)}]
    if family.family in FAMILY_INVALIDATION:
        plan.invalidation.append({"kind": "metric_crossed", "value": FAMILY_INVALIDATION[family.family]})
    return plan
