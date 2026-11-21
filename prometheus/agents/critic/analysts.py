"""Analysts: what the street's recommendation counts say (Critic v3, design/critic_v3_two_reasons_spec.md section 3, design choices entry 137). No network, no model.

    result = assess(symbol, periods, config.analysts)       # periods: the Data Clerk's answer, newest first (`DataClerk.request_analyst_ratings`)

Jacob: *"Use finnhub or massive.com analyst recommendations in the Critic Scoring."* Only Finnhub's `/stock/recommendation` is in the licence (monthly counts of strong buy, buy, hold, sell and strong sell for each company); price targets,
upgrades and downgrades and Massive's Benzinga ratings answer 403, so they are not used. The Critic reaches the counts only through the Data Clerk, which reads them POINT IN TIME (a month is usable from its own first day, and
nothing after the day the Time Machine is working as of).

    n     = strongBuy + buy + hold + sell + strongSell                        of the newest usable month
    mean  = (5 x strongBuy + 4 x buy + 3 x hold + 2 x sell + 1 x strongSell) / n        1 (everyone says strong sell) to 5 (strong buy)
    score = 100 x (mean - 1) / 4                                              the dimension `analysts`, 10 of 100 points

Not applicable (the dimension leaves the weighted average, as any dimension may) when there is no usable month or `n` is under `analysts.min_analysts`. The change of the mean since the month `analysts.trend_months` earlier is shown
(`mean_change`), never scored. The SIGN `analysts_bearish` (a WARNING, a sign of the group `news`: it denies only together with a bad sign in another group) fires when the mean is at or under `analysts.bearish_mean` (more sell than
buy) or has fallen by `analysts.bearish_drop` or more over those months.
"""
from dataclasses import dataclass, field
from datetime import date

from prometheus.agents.critic.common import warning
from prometheus.agents.critic.config import Analysts
from prometheus.agents.critic.record import AnalystsDetail, Finding, Measure

POINTS = {"strongBuy": 5, "buy": 4, "hold": 3, "sell": 2, "strongSell": 1}
WORDS = {"strongBuy": "strong buy", "buy": "buy", "hold": "hold", "sell": "sell", "strongSell": "strong sell"}


@dataclass
class AnalystsResult:
    findings: list[Finding] = field(default_factory=list)
    score: float = 0.0
    measures: list[Measure] = field(default_factory=list)
    detail: AnalystsDetail = field(default_factory=AnalystsDetail)
    applicable: bool = False


def counts_of(period: dict) -> dict[str, int] | None:
    """The five counts of one period, or None when one is missing, is not a whole number, or is negative (a bool is not a count)."""
    if not isinstance(period, dict):
        return None
    counts = {name: period.get(name) for name in POINTS}
    return counts if all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in counts.values()) else None


def mean_of(counts: dict[str, int]) -> float | None:
    """The mean rating, 1 to 5; None with no analyst."""
    n = sum(counts.values())
    return sum(POINTS[name] * c for name, c in counts.items()) / n if n else None


def month_of(period: str) -> tuple[int, int] | None:
    try:
        d = date.fromisoformat(period)
    except (TypeError, ValueError):
        return None
    return d.year, d.month


def months_before(period: str, months: int) -> tuple[int, int] | None:
    """The (year, month) that is `months` before a period's own month."""
    ym = month_of(period)
    if ym is None:
        return None
    index = ym[0] * 12 + (ym[1] - 1) - months
    return index // 12, index % 12 + 1


def _line(counts: dict[str, int]) -> str:
    return ", ".join(f"{counts[name]} {WORDS[name]}" for name in POINTS)


def assess(symbol: str, periods: list[dict] | None, config: Analysts) -> AnalystsResult:
    """The analysts' dimension and sign for one symbol. `periods` is what the Data Clerk gave (newest first; not trusted to be sorted); None or empty is "no usable period"."""
    out = AnalystsResult()
    usable = sorted(((p["period"], c) for p in (periods or []) if isinstance(p, dict) and isinstance(p.get("period"), str) and month_of(p["period"]) and (c := counts_of(p)) is not None), key=lambda x: x[0], reverse=True)
    if not usable:
        out.detail = AnalystsDetail(note="no analyst ratings the day could know")
        out.measures = [Measure(name="analyst_mean", x=None, value=0.0, weight=100, available=False, note=out.detail.note)]
        return out
    period, counts = usable[0]
    n = sum(counts.values())
    if n < config.min_analysts:
        out.detail = AnalystsDetail(period=period, n=n, counts=counts, note=f"only {n} analyst(s) in {period}; {config.min_analysts} are needed to score it")
        out.measures = [Measure(name="analyst_mean", x=None, value=0.0, weight=100, available=False, note=out.detail.note)]
        return out
    mean = mean_of(counts)
    earlier = months_before(period, config.trend_months)
    before = next(((p, c) for p, c in usable[1:] if month_of(p) == earlier and sum(c.values()) >= config.min_analysts), None)
    change = None if before is None else round(mean - mean_of(before[1]), 4)
    out.applicable = True
    out.score = round(100 * (mean - 1) / 4, 2)
    out.detail = AnalystsDetail(applicable=True, period=period, n=n, counts=counts, mean=round(mean, 4), score=out.score, mean_change=change, compared_period=before[0] if before else None,
                                note=f"{_line(counts)} in {period} ({n} analysts)")
    out.measures = [Measure(name="analyst_mean", x=round(mean, 3), value=round((mean - 1) / 4, 4), weight=100, available=True, note=out.detail.note),
                    Measure(name="mean_change", x=change, value=0.0, weight=0, available=change is not None,
                            note=f"not scored: the mean {'rose' if change and change > 0 else 'fell' if change and change < 0 else 'did not change'} {abs(change):.2f} since {before[0]}" if change is not None else
                            f"not scored: no month {config.trend_months} months earlier to compare with")]
    low, dropped = mean <= config.bearish_mean, change is not None and change <= config.bearish_drop
    if low or dropped:
        why = ([f"the mean rating is {mean:.2f} of 5 (1 is strong sell, 5 strong buy), at or under {config.bearish_mean:g}"] if low else []) + \
              ([f"the mean fell {abs(change):.2f} since {before[0]}, by {abs(config.bearish_drop):g} or more"] if dropped else [])
        out.findings.append(warning("analysts", "analysts_bearish", f"The analysts lean against {symbol}: {' and '.join(why)} ({_line(counts)} in {period}).", value=[round(mean, 4), change], threshold=[config.bearish_mean, config.bearish_drop]))
    return out
