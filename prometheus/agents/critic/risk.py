"""Risk fit: does the risk of this position match the user's profile? (Critic v1 spec section 8.4, step 7). No network, no model.

    result = assess(rec, view, fetched, price, config.risk)

The Critic measures the risk ITSELF, from the numbers it just had the Data Clerk calculate (spec P2) and the plan's stop, and compares it with what the user accepts. It judges the COMPANY against the user's
risk tolerance (age and horizon included), never how much to buy: it does not know the user's money, and the size of a purchase is the Account Manager's (agents/account_manager/sizing.py).

    effective tolerance   the lowest of the stated one, the cap for the user's age and the cap for their horizon
    risk score            0 to 100 (higher is riskier): eight measures, each `clip((x - safe) / (risky - safe), 0, 1)` times its weight; a measure that could not be made counts `unknown_points`
    risk level            low, medium, high or very_high from the score; very_high is allowed to nobody

A recommendation is denied when its level is above what the effective tolerance allows, or when the profile excludes the symbol, its sector or its price. A SELL (`reduce`) lowers risk and is not judged here.
"""
import math
from dataclasses import dataclass, field

from prometheus.agents.advisor.record import Recommendation
from prometheus.agents.critic.common import blocker, clip, info, metric_value
from prometheus.agents.critic.config import LEVELS, Risk, RiskMeasure
from prometheus.agents.critic.record import Finding, Measure, RiskDetail
from prometheus.common.user_profile import TOLERANCES, RiskView


@dataclass
class RiskResult:
    findings: list[Finding] = field(default_factory=list)
    score: float = 0.0                         # risk fit, 0 to 100
    measures: list[Measure] = field(default_factory=list)
    detail: RiskDetail = field(default_factory=RiskDetail)
    applicable: bool = True


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


def assess(rec: Recommendation, view: RiskView, fetched: dict | None, price: float | None, config: Risk) -> RiskResult:
    """The risk findings of one record, the risk score and level, and the risk-fit score."""
    out = RiskResult()
    stated = view.risk_tolerance
    effective, why = effective_tolerance(view, config)
    allowed = config.allowed[effective]
    out.detail = RiskDetail(tolerance_stated=stated, tolerance_effective=effective, allowed_level=allowed)
    action, trade = rec.action.action, rec.trade
    if action not in ("open", "add"):
        out.applicable = False
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

    # the risk score
    flags = flags_of(rec, fetched, config)
    inputs = {
        "stop_loss_pct": trade.stop_distance / trade.entry_mid if trade.stop_distance and trade.entry_mid else None,
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
