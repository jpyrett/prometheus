"""Risk fit: does the risk of this position match the user's profile? (Critic v1 spec section 8.4, step 7). No network, no model.

    result = assess(rec, view, fetched, price, equity, basis, config.risk)

The Critic measures the risk ITSELF, from the numbers it just had the Data Clerk calculate (spec P2), the plan's stop and the size the profile would allow, and compares it with what the user accepts:

    effective tolerance   the lowest of the stated one, the cap for the user's age and the cap for their horizon
    approved shares       min(what the Advisor suggested, what the position limit, the loss-per-trade limit and the order limit allow): the Critic may CUT a size, never raise it
    risk score            0 to 100 (higher is riskier): nine measures, each `clip((x - safe) / (risky - safe), 0, 1)` times its weight; a measure that could not be made counts `unknown_points`
    risk level            low, medium, high or very_high from the score; very_high is allowed to nobody

A recommendation is denied when its level is above what the effective tolerance allows, when not even one share fits the profile's limits, or when the profile excludes the symbol, its sector or
its price. A SELL (`reduce`) lowers risk and is not judged here.
"""
import math
from dataclasses import dataclass, field

from prometheus.agents.advisor.record import Recommendation
from prometheus.agents.critic.common import blocker, clip, info, metric_value, warning
from prometheus.agents.critic.config import LEVELS, Risk, RiskMeasure
from prometheus.agents.critic.record import Finding, Measure, RiskDetail
from prometheus.common.user_profile import TOLERANCES, RiskView


@dataclass
class RiskResult:
    findings: list[Finding] = field(default_factory=list)
    score: float = 0.0                         # risk fit, 0 to 100
    measures: list[Measure] = field(default_factory=list)
    detail: RiskDetail = field(default_factory=RiskDetail)
    approved_shares: int = 0
    applicable: bool = True
    price: float | None = None


def effective_tolerance(view: RiskView, config: Risk) -> tuple[str, str | None]:
    """(the tolerance the Critic applies, what lowered it: "age 72", "a 2-year horizon" or None). The lowest of the stated one and the caps."""
    rank = {name: i for i, name in enumerate(TOLERANCES)}
    best, why = view.risk_tolerance, None
    if view.age is not None:
        cap = next((c for c in config.age_caps if view.age >= c.min_age), None)   # the rows run from the oldest down: the first the age reaches
        if cap and rank[cap.tolerance] < rank[best]:
            best, why = cap.tolerance, f"age {view.age}"
    if view.investment_horizon_years is not None:
        cap = next((c for c in config.horizon_caps if view.investment_horizon_years < c.under_years), None)   # the rows run from the shortest up: the first the horizon is under
        if cap and rank[cap.tolerance] < rank[best]:
            best, why = cap.tolerance, f"a {view.investment_horizon_years:g}-year horizon"
    return best, why


def level_of(score: float, config: Risk) -> str:
    """low, medium, high or very_high: the first level whose upper bound the score is under."""
    for level in ("low", "medium", "high"):
        if score < config.levels[level]:
            return level
    return "very_high"


def measure_points(measure: RiskMeasure, x: float | None, unknown: float) -> tuple[float, bool]:
    """(points 0 to 1, whether the measure could be made). `flags` is already a share; every other measure is `clip((x - safe) / (risky - safe))`, of the absolute value or of log10 when it says so."""
    if x is None:
        return unknown, False
    if measure.name == "flags":
        return clip(x), True
    if measure.abs:
        x = abs(x)
    safe, risky = measure.safe, measure.risky
    if measure.log:
        if x <= 0:
            return unknown, False
        x, safe, risky = math.log10(x), math.log10(safe), math.log10(risky)
    return clip((x - safe) / (risky - safe)), True


def flags_of(rec: Recommendation, fetched: dict | None, config: Risk) -> list[str]:
    """What makes this position riskier than its numbers say: a stressed or unknown market, a strategy conflict, a stockholders' deficit, a report inside the horizon, weak quality."""
    flags = []
    if rec.identity.regime in ("stress", "unknown"):
        flags.append(f"market regime {rec.identity.regime}")
    if rec.ranking.conflict:
        flags.append("strategies disagree")
    if metric_value(fetched, "negative_equity") is True:
        flags.append("stockholders' deficit")
    if rec.trade.gap_risk:
        flags.append("a report inside the horizon")
    if rec.quality.multiplier < config.weak_quality:
        flags.append(f"weak quality ({rec.quality.multiplier:.2f})")
    return flags


def size_caps(rec: Recommendation, view: RiskView, price: float | None, equity: float | None) -> dict[str, int | None]:
    """The most shares each limit of the profile allows, None for a limit that cannot be worked out (no price, no equity, no stop distance). The exposure cap is filled in by the review, over all its records."""
    caps: dict[str, int | None] = {"position": None, "loss": None, "order": None, "exposure": None}
    if not price or price <= 0:
        return caps
    held_value = rec.action.held_shares * price
    if equity:
        caps["position"] = max(0, math.floor((view.max_position_pct * equity - held_value) / price))
        if rec.trade.stop_distance and rec.trade.stop_distance > 0:
            caps["loss"] = max(0, math.floor(view.max_loss_per_trade_pct * equity / rec.trade.stop_distance))
    if view.max_order_dollars:
        caps["order"] = math.floor(view.max_order_dollars / price)
    return caps


def assess(rec: Recommendation, view: RiskView, fetched: dict | None, price: float | None, equity: float | None, equity_basis: str, config: Risk) -> RiskResult:
    """The risk findings of one record, its size under the profile, the risk score and level, and the risk-fit score."""
    out = RiskResult(price=price)
    stated = view.risk_tolerance
    effective, why = effective_tolerance(view, config)
    allowed = config.allowed[effective]
    out.detail = RiskDetail(tolerance_stated=stated, tolerance_effective=effective, allowed_level=allowed, equity=equity, equity_basis=equity_basis)
    action, trade = rec.action.action, rec.trade
    if action not in ("open", "add"):
        out.applicable = False
        out.approved_shares = trade.suggested_shares or rec.action.held_shares
        return out
    if why:
        out.findings.append(info("risk_fit", "tolerance_capped", f"Risk tolerance {stated} is lowered to {effective} by {why}.", value=effective))

    # the profile's own lists
    if rec.symbol in view.excluded_symbols:
        out.findings.append(blocker("risk_fit", "symbol_excluded", f"{rec.symbol} is on the user's list of excluded symbols."))
    if rec.identity.sector_etf and rec.identity.sector_etf in view.excluded_sectors:
        out.findings.append(blocker("risk_fit", "sector_excluded", f"{rec.symbol}'s sector ({rec.identity.sector_etf}) is on the user's list of excluded sectors."))
    if price is not None and price < view.min_price:
        out.findings.append(blocker("risk_fit", "price_below_profile_min", f"The price {price:.2f} is under the user's minimum of {view.min_price:g}.", value=price, threshold=view.min_price))

    # the size the profile allows
    caps = size_caps(rec, view, price, equity)
    known = {name: n for name, n in caps.items() if n is not None}
    needed_missing = price is None or equity is None or trade.stop_distance is None
    out.approved_shares = 0 if needed_missing else max(0, min([trade.suggested_shares, *known.values()]))
    out.detail.caps, out.detail.allowed_shares = caps, out.approved_shares
    if needed_missing:
        out.findings.append(blocker("risk_fit", "size_above_profile", "The position cannot be sized under the profile: " + ", ".join(
            reason for reason, missing in (("there is no price", price is None), ("the equity is unknown", equity is None), ("the plan has no stop distance", trade.stop_distance is None)) if missing) + "."))
    elif out.approved_shares == 0:
        limits = ", ".join(f"{name} {n}" for name, n in known.items())
        out.findings.append(blocker("risk_fit", "size_above_profile", f"Not even one share fits the user's limits (shares allowed: {limits}).", value=caps, threshold=1))
    elif out.approved_shares < trade.suggested_shares:
        tightest = min(known, key=known.get)
        out.findings.append(warning("risk_fit", "size_reduced", f"The size is cut from {trade.suggested_shares} to {out.approved_shares} shares by the user's {tightest} limit.",
                                    value=out.approved_shares, threshold=trade.suggested_shares))

    # the risk score
    held_value = rec.action.held_shares * (price or 0.0)
    flags = flags_of(rec, fetched, config)
    inputs = {
        "stop_loss_pct": trade.stop_distance / trade.entry_mid if trade.stop_distance and trade.entry_mid else None,
        "position_pct": (held_value + out.approved_shares * price) / equity if price and equity else None,
        "flags": min(1.0, config.flag_step * len(flags)),
    }
    total = 0.0
    for m in config.measures:
        x = inputs[m.name] if m.name in inputs else metric_value(fetched, m.name)
        points, available = measure_points(m, x, config.unknown_points)
        total += m.weight * points
        out.measures.append(Measure(name=m.name, x=None if x is None else round(x, 6), value=round(points, 4), weight=m.weight, available=available,
                                    note=None if available else "could not be measured: counted as unknown"))
    score = round(total, 1)
    level = level_of(score, config)
    out.detail = out.detail.model_copy(update={"risk_score": score, "risk_level": level, "flags": flags})
    if LEVELS.index(level) > LEVELS.index(allowed):
        out.findings.append(blocker("risk_fit", "risk_above_profile", f"The risk is {level} ({score:g}), above the {allowed} level a {effective} profile accepts.", value=score, threshold=config.levels[allowed]))
    blocked = any(f.severity == "blocker" for f in out.findings)
    out.score = 0.0 if blocked else round(clip(100 * (1 - 0.5 * score / config.levels[allowed]), 0.0, 100.0), 2)
    return out


def exposure_cap(running_dollars: float, price: float | None, equity: float | None, limit_pct: float) -> int | None:
    """The most shares that still fit under the user's limit on what ONE review may add, given what the earlier approvals of the same review already add. None when it cannot be worked out."""
    if not price or not equity:
        return None
    return max(0, math.floor((limit_pct * equity - running_dollars) / price))
