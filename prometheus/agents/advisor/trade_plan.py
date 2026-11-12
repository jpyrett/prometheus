"""The trade plan: where to enter, where the idea is wrong, how much, and when to stop waiting (Advisor v1 spec section 10, design choices entry 51).

    plan = build(family, metrics, quote, regime, equity, portfolio, today, config)      # a TradePlan

All arithmetic, no model. The plan is the part of a recommendation a Critic can check with a calculator, so every number comes with its basis:

    entry_zone      the last price +- 0.5 ATR; the last price is the quote when the order has one, else the last stored close (`entry_basis` says which)
    stop_distance   2 ATR for reversion and event, 3 ATR for trend and capital_return (or the distance to the 50-day mean when that is nearer and below the price); at least the typical
                    earnings move when a report falls inside the horizon (`gap_risk`; the earnings dates are not available in this build, so it never fires)
    stop_price      the entry mid minus the stop distance (long only: a bearish call on a name that is not held is `avoid`, not a short)
    target_price    the entry mid x (1 + 1.5 x the family's edge at its horizon); none for capital_return (it is held for the payout, not for a target)
    time_stop_date  `horizon_bars` weekdays after today (the calendar of the future has no holidays in it: the note says so)
    size            WHOLE shares: floor(risk budget / stop distance), where the risk budget is 1% of equity (0.5% when the regime is stress or unknown), then no more than 20% of
                    equity and 5% of the stock's average daily dollar volume. When that is 0 the plan says ONE share, `oversized: true`, and how much of the account one share is
    order_style     limit at the entry mid for reversion, for capital_return, for an event that has moved less than one ATR, and for any stock with a spread over 10 bps; a marketable limit
                    at the last price + 0.25 ATR for trend while it is above its 200-day mean; a limit otherwise
    invalidation    the rules that say "this idea is wrong or finished": the price under the stop, the time stop, a bearish event, and the family's own condition

TWO READINGS THE SPEC LEAVES OPEN, flagged in design choices entry 51: (1) the spec writes the share count as `floor(risk_budget x equity / stop_distance)`, which multiplies by equity twice (the budget is
already 1% OF equity): it is `floor(risk_budget / stop_distance)` here; (2) "the family's edge at its horizon": the 20-day edge is taken as it is for a horizon of 20 bars or less (the
reversion edge is realised inside its 10 bars, spec 8.5) and scaled up in proportion for a longer one (trend 63 bars = 3.15 x), so a long-horizon target is not smaller than its own stop.
"""
import math
from datetime import date, timedelta

from prometheus.agents.advisor.config import AdvisorConfig
from prometheus.agents.advisor.events import BEARISH
from prometheus.agents.advisor.gates import value
from prometheus.agents.advisor.record import FamilyResult, TradePlan
from prometheus.common.portfolio import Portfolio
from prometheus.tools import trading_math as tm

FAMILY_INVALIDATION = {"trend": "below_sma_200", "reversion": "rsi_14_above_70"}   # capital_return and conviction have none of their own: the stop, the time and a bearish event end them


def add_weekdays(day: date, n: int) -> date:
    """The date `n` weekdays after `day` (holidays are not known for a future date)."""
    while n > 0:
        day += timedelta(days=1)
        n -= day.weekday() < 5
    return day


def price_of(symbol: str, metrics: dict | None, quote: dict | None) -> tuple[float | None, str]:
    """(the price, where it came from): the order's quote, else the last stored close."""
    if quote and isinstance(quote.get("price"), (int, float)) and quote["price"] > 0:
        return float(quote["price"]), "quote"
    last = value(metrics, "last_close")
    return (float(last), "last_close") if last else (None, "none")


def equity_of(portfolio: Portfolio, metrics_by_symbol: dict[str, dict], quotes: dict[str, dict]) -> tuple[float | None, str, dict[str, float]]:
    """(equity, basis, prices): cash plus every holding at its price (a quote when the order has one, else the last stored close). None when a holding has no price: the size of a position is
    never computed from a guess. `basis` says what each price was."""
    prices, bases = {}, set()
    for position in portfolio.positions:
        price, basis = price_of(position.symbol, metrics_by_symbol.get(position.symbol), quotes.get(position.symbol))
        if price is None:
            return None, f"no price for {position.symbol}", prices
        prices[position.symbol] = price
        bases.add(basis)
    return portfolio.cash_avail + sum(p.shares * prices[p.symbol] for p in portfolio.positions), (", ".join(sorted(bases)) or "cash only"), prices


def funds_check(cost: float, portfolio: Portfolio, prices: dict[str, float], today: date) -> tuple[bool, str]:
    """Whether the money for `cost` exists: free cash, or free cash plus what could be sold from the holdings that are in profit and sellable today (RaiseFunds' default rules: only holdings in
    the green, and one share of every holding is kept). A FLAG for the Account Manager: it does not run the tree of thoughts and it plans no sale."""
    if portfolio.cash_avail >= cost:
        return True, f"cash ${portfolio.cash_avail:,.2f} covers ${cost:,.2f}"
    holdings = [{"symbol": p.symbol, "shares": max(0, portfolio.sellable_shares(p.symbol, today) - 1), "price": prices[p.symbol], "basis": p.cost_basis} for p in portfolio.positions if p.symbol in prices]
    raisable = sum(h["shares"] * h["price"] for h in tm.rank_green(holdings, only_green=True))
    shortfall = cost - portfolio.cash_avail
    return raisable >= shortfall, f"short by ${shortfall:,.2f}; ${raisable:,.2f} could be raised from holdings in profit that are sellable today (one share of each kept)"


def build(family: FamilyResult | None, metrics: dict | None, quote: dict | None, regime: str, equity: tuple[float | None, str, dict[str, float]], portfolio: Portfolio, today: date,
          config: AdvisorConfig, symbol: str = "", reaction_atr: float | None = None, sizing: bool = True) -> TradePlan:
    """The plan for one symbol on `family` (None for a name that is not being opened: a reduction or an avoidance has an entry zone and no size). `equity` is `equity_of(...)`."""
    tc = config.trade
    last, basis = price_of(symbol, metrics, quote)
    atr, sma50, adv, spread = value(metrics, "atr_14"), value(metrics, "sma_50"), value(metrics, "adv_20d_dollars"), value(metrics, "median_spread_bps_1d")
    plan = TradePlan(family=family.family if family else None, entry_basis=basis, horizon_bars=family.horizon_bars if family else None)
    if last is None or atr is None:
        plan.notes.append("no plan prices: " + ("there is no price" if last is None else "atr_14 is missing (needs 20 bars)"))
        return plan
    mid = last
    plan.entry_mid, plan.entry_zone = round(mid, 4), [round(last - tc.entry_atr * atr, 4), round(last + tc.entry_atr * atr, 4)]
    if family is None:
        return plan
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
    total, equity_basis, _ = equity
    plan.equity, plan.equity_basis = total, equity_basis
    if sizing and total:
        budget = (tc.risk_pct_stress if regime in ("stress", "unknown") else tc.risk_pct) * total
        plan.risk_budget = round(budget, 2)
        shares = math.floor(budget / distance)
        shares = min(shares, math.floor(tc.max_position_pct * total / mid), math.floor(tc.max_adv_pct * adv / mid) if adv else shares)
        if shares <= 0:
            plan.suggested_shares, plan.oversized = 1, True
            plan.size_note = f"one share is {mid / total:.0%} of equity; the {budget / total:.1%} risk budget allows 0"
        else:
            plan.suggested_shares = shares
        plan.position_pct = round(plan.suggested_shares * mid / total, 4)
        plan.funds_feasible, plan.funds_note = funds_check(plan.suggested_shares * mid, portfolio, equity[2], today)
    elif sizing:
        plan.notes.append(f"no size: equity is unknown ({equity_basis})")
    style = "limit"
    if family.family == "trend" and value(metrics, "above_sma_200") is True and not (spread is not None and spread > tc.limit_spread_bps):
        style = "marketable_limit"
    elif family.family == "event" and (reaction_atr is None or reaction_atr >= tc.event_limit_reaction_atr):
        plan.notes.append("order style: a limit at the entry mid (the move since the event is one ATR or more, or unknown)")
    plan.order_style = style
    plan.limit_price = round(last + tc.trend_limit_atr * atr, 4) if style == "marketable_limit" else round(mid, 4)
    plan.invalidation = [{"kind": "price_below", "value": plan.stop_price}, {"kind": "days_elapsed", "value": family.horizon_bars}, {"kind": "event_class", "value": list(BEARISH)}]
    if family.family in FAMILY_INVALIDATION:
        plan.invalidation.append({"kind": "metric_crossed", "value": FAMILY_INVALIDATION[family.family]})
    return plan
