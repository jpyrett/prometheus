"""Timing: has the recommendation expired? (Critic v1 spec section 8.3, step 6). No network, no model.

    result = assess(rec, now, quote, last_close, record_date, config.timing)

A recommendation is for the trading day it was made on, and it can only be acted on inside that day's trading window (pre-market, regular hours, and after-hours when the next day also trades: see
`common/market_calendar.py`). So the Critic asks four questions: is the market open now; is the record from today's trading day; is the price still where the plan assumed; and are the prices it is
looking at recent. The price it finds is the DEAL PRICE the risk and value checks use: the live quote when it is usable, else the re-fetched last close, else nothing.

An approval does not last: `valid_until` is the earlier of the end of the trading window and a short time to live after the review. After it the Account Manager must not trade (and
`work_order.verify_trade` refuses): the price check is old.
"""
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from prometheus.agents.advisor.record import Recommendation
from prometheus.agents.critic.common import blocker, et_text, info, utc_text, warning
from prometheus.agents.critic.config import Timing
from prometheus.agents.critic.record import Finding, Measure, TimingDetail
from prometheus.common.market_calendar import Session, eastern_date, previous_trading_day, session_at
from prometheus.common.work_order import parse_utc


@dataclass
class TimingResult:
    findings: list[Finding] = field(default_factory=list)
    score: float = 0.0
    measures: list[Measure] = field(default_factory=list)
    detail: TimingDetail | None = None
    deal_price: float | None = None
    price_basis: str | None = None
    session: Session | None = None


def record_day(rec: Recommendation, now: datetime) -> date | None:
    """The trading day a record is for: the date the Advisor was given (a stamp that ends in T00:00:00Z is that date as it is), else the stamp's date in US Eastern. None when unreadable."""
    stamp = rec.identity.as_of_utc
    try:
        if stamp.endswith("T00:00:00Z"):
            return date.fromisoformat(stamp[:10])
        moment = parse_utc(stamp)
        return eastern_date(moment) if moment else None
    except ValueError:
        return None


def zone_factor(price: float | None, zone: list[float] | None, mid: float | None, config: Timing) -> float:
    """1.0 at or under the entry mid (and not under the zone), falling in a straight line to `top_of_zone_factor` at the top of the zone, 0 above it; `below_zone_factor` under the zone; 0 with no price."""
    if price is None or not zone or mid is None:
        return 0.0
    bottom, top = zone
    if price < bottom:
        return config.below_zone_factor
    if price <= mid:
        return 1.0
    if price > top:
        return 0.0
    return 1.0 - (1.0 - config.top_of_zone_factor) * ((price - mid) / (top - mid)) if top > mid else config.top_of_zone_factor


def market_hours_ignored() -> bool:
    """`IGNORE_MARKET_HOURS` in the environment or `.env` (true, yes, on or 1): the closed market is not a blocker. Read when a review runs, so a test can set it for one review."""
    from prometheus.rag.config import setting
    return setting("IGNORE_MARKET_HOURS", "false").strip().lower() in ("1", "true", "yes", "on")


def assess(rec: Recommendation, now: datetime, quote: dict | None, last_close: dict | None, config: Timing, calendar=None, ignore_market_hours: bool = False) -> TimingResult:
    """Every timing finding of one record, the score, and the deal price. `quote` is the Data Clerk's quote (or None), `last_close` the re-fetched `last_close` metric result."""
    out = TimingResult()
    session = out.session = session_at(now, calendar)
    trade = rec.trade
    today = eastern_date(now, calendar)
    day = record_day(rec, now)

    if session.state == "closed" and ignore_market_hours:
        out.findings.append(info("timing", "market_hours_ignored", f"The market is closed ({session.reason}); IGNORE_MARKET_HOURS is set, so that does not deny the recommendation. The next window opens {et_text(session.next_open_utc)}.", value=session.reason))
    elif session.state == "closed":
        out.findings.append(blocker("timing", "market_closed", f"The market is closed ({session.reason}); the next trading window opens {et_text(session.next_open_utc)}.", value=session.reason))
    elif session.state != "regular":
        out.findings.append(info("timing", "extended_hours", f"It is the {session.state.replace('_', '-')} session: fewer buyers and sellers, wider spreads.", value=session.state))
    want = session.trading_date if session.is_open else session.last_trading_date
    if day is None or day != want:
        out.findings.append(blocker("timing", "record_expired", f"The recommendation is for {day or 'an unreadable date'}, but the trading day it can be acted on is {want}.", value=str(day), threshold=str(want)))
    if trade.time_stop_date and date.fromisoformat(trade.time_stop_date) < today:
        out.findings.append(blocker("timing", "time_stop_passed", f"The time stop ({trade.time_stop_date}) has passed.", value=trade.time_stop_date))

    # the stored bars
    bar_day = date.fromisoformat(last_close["as_of"]) if last_close and last_close.get("as_of") and last_close.get("value") is not None else None
    needed = previous_trading_day(session.last_trading_date, calendar)
    if bar_day is None:
        out.findings.append(blocker("timing", "bars_stale", f"There are no stored price bars for {rec.symbol}.", threshold=str(needed)))
    elif bar_day < needed:
        out.findings.append(blocker("timing", "bars_stale", f"The newest stored bar is {bar_day}; at least the session of {needed} is needed.", value=str(bar_day), threshold=str(needed)))

    # the quote
    quote_price, quoted_at = (quote or {}).get("price"), parse_utc((quote or {}).get("as_of_utc"))
    usable = bool(quote_price)
    if not quote_price:
        out.findings.append(blocker("timing", "no_live_price", f"No current quote came back for {rec.symbol}."))
    else:
        quote_day = eastern_date(quoted_at, calendar) if quoted_at else None
        age = (now - quoted_at).total_seconds() / 60 if quoted_at else None
        too_old = quote_day is None or quote_day not in (session.last_trading_date, previous_trading_day(session.last_trading_date, calendar)) \
            or (session.state == "regular" and age is not None and age > config.max_quote_age_minutes)
        if too_old:
            usable = False
            out.findings.append(blocker("timing", "quote_stale", f"The quote of {rec.symbol} is from {et_text(quoted_at)}: too old to trade on" +
                                        (f" (more than {config.max_quote_age_minutes:g} minutes during the session)." if session.state == "regular" else "."), value=utc_text(quoted_at), threshold=config.max_quote_age_minutes))
    last = last_close.get("value") if last_close else None
    out.deal_price, out.price_basis = (float(quote_price), "quote") if usable else (float(last), "last_close") if last else (None, None)
    price = out.deal_price

    # the price against the plan
    zone = trade.entry_zone
    position = None
    if zone and price is not None and zone[1] > zone[0]:
        position = (price - zone[0]) / (zone[1] - zone[0])
    if price is not None and zone and price > zone[1]:
        out.findings.append(blocker("timing", "price_left_zone", f"The price {price:.2f} is above the top of the entry zone ({zone[1]:.2f}): buying now would be chasing it.", value=price, threshold=zone[1]))
    if trade.stop_price is not None and price is not None and price <= trade.stop_price:
        out.findings.append(blocker("timing", "stop_already_hit", f"The price {price:.2f} is at or under the stop ({trade.stop_price:.2f}): the idea is already wrong.", value=price, threshold=trade.stop_price))
    elif price is not None and zone and price < zone[0]:
        out.findings.append(warning("timing", "price_below_zone", f"The price {price:.2f} is under the entry zone ({zone[0]:.2f} to {zone[1]:.2f}): it has fallen since the plan was made.", value=price, threshold=zone[0]))

    session_part = config.session_factor.get(session.state, 0.0)
    zone_part = zone_factor(price, zone, trade.entry_mid, config)
    out.measures = [Measure(name="session", x=session.state, value=session_part, weight=50), Measure(name="zone", x=None if position is None else round(position, 4), value=round(zone_part, 4), weight=50,
                                                                                                    available=price is not None)]
    out.score = round(100 * (0.5 * session_part + 0.5 * zone_part), 2)
    out.detail = TimingDetail(session=session.state, trading_date=str(session.trading_date) if session.trading_date else None, session_reason=session.reason,
                              window_closes_utc=utc_text(session.window_closes_utc), next_open_utc=utc_text(session.next_open_utc), record_date=str(day) if day else None,
                              quote_price=float(quote_price) if quote_price else None, quote_as_of_utc=utc_text(quoted_at), deal_price=price, price_basis=out.price_basis,
                              zone_position=None if position is None else round(position, 4))
    return out


def valid_until(session: Session, now: datetime, ttl_minutes: float) -> datetime:
    """When an approval made at `now` stops being valid: the earlier of the end of today's trading window and `ttl_minutes` after the review."""
    expiry = now + timedelta(minutes=ttl_minutes)
    return min(expiry, session.window_closes_utc) if session.window_closes_utc else expiry
