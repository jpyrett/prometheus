"""Chart: what the shape of the price says (Critic v1 Amendment 1, design choices entry 62, step 8b). No network, no model.

    result = assess(rec, fetched, config.chart)

Twenty-three simple patterns, each a plain statement over numbers the Data Clerk calculated for the Critic (`CHART_METRICS`): the 50-day mean against the 200-day mean (a golden or a death cross), the
stacking of price and means, the highs and lows of three windows, a breakout on volume, the distance to the 52-week high, MACD, RSI and on-balance volume, and the dead cat bounce; and, since version 3, the eight of
`patterns.py` (double top and bottom, head and shoulders and its inverse, the triangles, the 52-week breakout and breakdown), which read the closing prices the Data Clerk gives (`close_history`). Every pattern is
arithmetic on those numbers and its thresholds are in `configs/critic_v1.json` (`chart`). A pattern whose number could not be made is `unknown`: it is not judged, never guessed.

    score = clip(neutral + the points of the patterns that hold, 0, 100)           neutral 50: a chart with no pattern is neither good nor bad
    pattern_bonus = clamp(the same net points x per_point, -max_points, +max_points)    ALSO a bonus on the review's final score (Critic v3): a bullish chart raises it, a bearish one lowers it

The dimension is a weaker voice than the others (10 of 100 points). One pattern is a BLOCKER: the dead cat bounce (a fall of at least 25% from a peak, a small bounce off the low in the last
sessions, the price still far under the peak and under its 50-day mean) is the pattern that most often precedes a second leg down. The golden cross and the rest only add points; a death cross
is a warning. Version 3 groups the signs: the dead cat bounce, `trend_not_confirmed`, `chart_bearish` (the net points at or under `chart.bearish_at`) and `poor_performance` are signs of the group `performance`
(`reasons.py`): one of them does not deny, two groups with a bad sign do.
"""
from dataclasses import dataclass, field

from prometheus.agents.advisor.record import Recommendation
from prometheus.agents.critic.common import blocker, clip, info, metric_value, pct, warning
from prometheus.agents.critic import patterns
from prometheus.agents.critic.config import PATTERNS, Chart, Performance
from prometheus.agents.critic.record import ChartDetail, Finding, Measure, PatternHit

CHART_METRICS = ("sma_200", "sma_50_vs_200", "cross_50_200_20d", "structure_63d", "breakout_20d", "volume_ratio_1d", "macd_hist", "rsi_14", "obv_20d_change",
                 "decline_from_peak_63d", "rebound_from_trough_63d", "sessions_since_trough_63d", "close_history")   # asked for with `sma_50`, `last_close` and `pct_of_52w_high`, which the review already has
FACTS = ("last_close", "sma_50", "pct_of_52w_high", *CHART_METRICS)
SERIES = ("close_history",)      # a series of closes, not a number: the swing patterns read it, and it is not kept in the review's facts
TREND_METRICS = ("ret_5d",)      # asked for with the review for rule C5 (`trend_not_confirmed`); the Critic's own reading, never compared with the record's (`recheck.ASKED_ONLY`)
PERFORMANCE_METRICS = ("ret_63d",)   # asked for with the review for `poor_performance` (Critic v3), next to `ret_21d` (which it always asks); like every `OWN` name it is compared only when the record holds it too

Judgement = tuple[bool | None, str]      # (does the pattern hold: None when a number it needs is missing, the numbers in words)
UNKNOWN: Judgement = (None, "")


@dataclass
class ChartResult:
    findings: list[Finding] = field(default_factory=list)
    score: float = 0.0
    measures: list[Measure] = field(default_factory=list)
    detail: ChartDetail = field(default_factory=ChartDetail)
    applicable: bool = True


def _when(age: int) -> str:
    return "today" if age == 1 else f"{age - 1} sessions ago"


def _cross(f: dict, golden: bool) -> Judgement:
    cross = f["cross_50_200_20d"]
    if cross is None:
        return UNKNOWN
    held = cross > 0 if golden else cross < 0
    return held, (f"the 50-day mean crossed {'above' if golden else 'under'} the 200-day mean {_when(abs(cross))}" if held else "no crossing of the 50- and 200-day means in the last 20 sessions" if cross == 0
                  else f"the last crossing was {'a death' if cross < 0 else 'a golden'} cross")


def _alignment(f: dict, bullish: bool) -> Judgement:
    close, fast, slow = f["last_close"], f["sma_50"], f["sma_200"]
    if None in (close, fast, slow):
        return UNKNOWN
    held = close > fast > slow if bullish else close < fast < slow
    return held, f"close {close:.2f}, 50-day mean {fast:.2f}, 200-day mean {slow:.2f}"


def _structure(f: dict, want: str) -> Judgement:
    shape = f["structure_63d"]
    return UNKNOWN if shape is None else (shape == want, f"the last three windows of 21 sessions show {'higher highs and higher lows' if shape == 'up' else 'lower highs and lower lows' if shape == 'down' else 'no clear rise or fall'}")


def _breakout(f: dict, c: Chart) -> Judgement:
    broke, volume = f["breakout_20d"], f["volume_ratio_1d"]
    if broke is False:
        return False, "the close is not above the highest high of the 20 sessions before"
    if broke is None or volume is None:
        return UNKNOWN
    return volume >= c.volume_ratio_min, f"a 20-session high on {volume:.1f} times the usual volume (it takes {c.volume_ratio_min:g})"


def _near_high(f: dict, c: Chart) -> Judgement:
    high = f["pct_of_52w_high"]
    return UNKNOWN if high is None else (high >= c.near_high, f"{pct(high, 0)} of the 52-week high (it takes {pct(c.near_high, 0)})")


def _macd(f: dict, positive: bool) -> Judgement:
    hist = f["macd_hist"]
    return UNKNOWN if hist is None else ((hist > 0) if positive else (hist < 0), f"the MACD histogram is {hist:+.3f}")


def _healthy_rsi(f: dict, c: Chart) -> Judgement:
    rsi = f["rsi_14"]
    return UNKNOWN if rsi is None else (c.rsi_healthy_low <= rsi <= c.rsi_healthy_high, f"RSI {rsi:.1f} (healthy is {c.rsi_healthy_low:g} to {c.rsi_healthy_high:g})")


def _overbought(f: dict, c: Chart) -> Judgement:
    rsi = f["rsi_14"]
    return UNKNOWN if rsi is None else (rsi >= c.rsi_overbought, f"RSI {rsi:.1f} (overbought from {c.rsi_overbought:g})")


def _obv(f: dict, rising: bool) -> Judgement:
    obv = f["obv_20d_change"]
    return UNKNOWN if obv is None else ((obv > 0) if rising else (obv < 0), f"on-balance volume changed {obv:+.2f} of the 20-session volume")


def _dead_cat(f: dict, c: Chart) -> Judgement:
    """A fall of at least `min_decline` from the 63-session peak close to the lowest close after it, a bounce of at least `min_rebound` off that low that began no more than `max_trough_age` sessions
    ago, the price still at least `max_vs_peak` under the peak, and under its 50-day mean. The distance to the peak is `(1 + decline) x (1 + rebound) - 1`."""
    decline, rebound, age, fast, close = (f[n] for n in ("decline_from_peak_63d", "rebound_from_trough_63d", "sessions_since_trough_63d", "sma_50", "last_close"))
    if None in (decline, rebound, age, fast, close):
        return UNKNOWN
    vs_peak = round((1 + decline) * (1 + rebound) - 1, 6)
    d = c.dead_cat
    held = decline <= d.min_decline and rebound >= d.min_rebound and age <= d.max_trough_age and vs_peak <= d.max_vs_peak and close < fast
    return held, (f"it fell {pct(abs(decline), 0)} from its peak, bounced {pct(rebound, 0)} off the low {age} sessions ago, is still {pct(abs(vs_peak), 0)} under the peak"
                  f" and {'under' if close < fast else 'over'} its 50-day mean")


JUDGES = {
    "golden_cross": lambda f, c: _cross(f, True), "death_cross": lambda f, c: _cross(f, False),
    "bull_alignment": lambda f, c: _alignment(f, True), "bear_alignment": lambda f, c: _alignment(f, False),
    "uptrend_structure": lambda f, c: _structure(f, "up"), "downtrend_structure": lambda f, c: _structure(f, "down"),
    "breakout_on_volume": _breakout, "near_high": _near_high,
    "macd_positive": lambda f, c: _macd(f, True), "macd_negative": lambda f, c: _macd(f, False),
    "healthy_rsi": _healthy_rsi, "overbought": _overbought,
    "obv_rising": lambda f, c: _obv(f, True), "obv_falling": lambda f, c: _obv(f, False),
    "dead_cat_bounce": _dead_cat,
    **patterns.JUDGES,
}
assert set(JUDGES) == set(PATTERNS)


def trend_confirmation(rec: Recommendation, ret_5d: float | None, config: Chart) -> list[Finding]:
    """Rule C5: `trend_not_confirmed` (a BLOCKER) for a TREND recommendation whose stock's return over the last five sessions, as the Critic's own metric gives it, is at or under `config.trend_confirm_ret_5d`; the warning
    `trend_unconfirmed_unknown` when that return cannot be had; nothing for any other family."""
    if rec.trade.family != "trend":
        return []
    if ret_5d is None:
        return [warning("chart", "trend_unconfirmed_unknown", f"The strategy is trend, which needs the stock to be rising, but {rec.symbol}'s return over the last five sessions could not be calculated: the move is not confirmed or denied.")]
    if ret_5d <= config.trend_confirm_ret_5d:
        return [blocker("chart", "trend_not_confirmed", f"The strategy is trend, which assumes a move goes on, but {rec.symbol}'s return over the last five sessions is {ret_5d:+.1%} (it needs more than {config.trend_confirm_ret_5d:+.1%}): "
                        f"the move is not confirmed.", value=ret_5d, threshold=config.trend_confirm_ret_5d)]
    return []


def poor_performance(rec: Recommendation, ret_63d: float | None, ret_21d: float | None, config: Performance) -> list[Finding]:
    """`poor_performance` (a WARNING, a sign of the group `performance`: it denies only with a bad sign in another group, rule V4): the stock's 63-session return, as the Critic's own re-asked metric gives it, is at or under
    `config.poor_ret_63d` AND its 21-session return is negative (a stock that is down on the quarter and still falling). Nothing when either return cannot be had."""
    if ret_63d is None or ret_21d is None or ret_63d > config.poor_ret_63d or ret_21d >= 0:
        return []
    return [warning("chart", "poor_performance", f"{rec.symbol} is down {abs(ret_63d):.1%} over 63 sessions (the limit is {config.poor_ret_63d:+.1%}) and {abs(ret_21d):.1%} over the last 21: poor performance.", value=[ret_63d, ret_21d], threshold=config.poor_ret_63d)]


def pattern_bonus(net_points: float | None, config) -> float:
    """`clamp(net x per_point, -max_points, +max_points)`, to two places: the net points of the chart patterns that hold, as a bonus on the final score (a bearish chart lowers it). Nothing without a net."""
    if not net_points:
        return 0.0
    return round(max(-config.max_points, min(config.max_points, net_points * config.per_point)), 2)


def assess(rec: Recommendation, fetched: dict | None, config: Chart) -> ChartResult:
    """The chart findings of one record, its score (0 to 100) and the detail."""
    out = ChartResult()
    facts = {name: metric_value(fetched, name) for name in FACTS}
    hits, unknown, measures, total = [], [], [], 0.0
    for p in config.patterns:
        held, text = JUDGES[p.name](facts, config)
        measures.append(Measure(name=p.name, x=p.points, value=1.0 if held else 0.0, weight=abs(p.points), available=held is not None, note=text or "could not be judged: a number it needs is missing"))
        if held is None:
            unknown.append(p.name)
        elif held:
            hits.append(PatternHit(name=p.name, direction=PATTERNS[p.name], points=p.points, evidence=text))
            total += p.points
    out.score = round(clip(config.neutral + total, 0.0, 100.0), 2)
    out.measures = measures
    out.detail = ChartDetail(patterns=hits, unknown=unknown, points=total, facts={k: v for k, v in facts.items() if v is not None and k not in SERIES})

    held = {h.name: h for h in hits}
    if "dead_cat_bounce" in held:
        out.findings.append(blocker("chart", "dead_cat_bounce", f"A dead cat bounce: {held['dead_cat_bounce'].evidence}.", value=[facts["decline_from_peak_63d"], facts["rebound_from_trough_63d"]],
                                    threshold=[config.dead_cat.min_decline, config.dead_cat.min_rebound]))
    if "death_cross" in held:
        out.findings.append(warning("chart", "death_cross", f"Death cross: {held['death_cross'].evidence}.", value=facts["cross_50_200_20d"]))
    if "golden_cross" in held:
        out.findings.append(info("chart", "golden_cross", f"Golden cross: {held['golden_cross'].evidence}.", value=facts["cross_50_200_20d"]))
    out.findings.extend(trend_confirmation(rec, metric_value(fetched, "ret_5d"), config))
    if total <= config.bearish_at:
        out.findings.append(warning("chart", "chart_bearish", f"The chart is bearish: the patterns that hold ({', '.join(h.name.replace('_', ' ') for h in hits if h.points < 0)}) net {total:+g} points (the sign is at {config.bearish_at:+g} or under).",
                                    value=total, threshold=config.bearish_at))
    if len(unknown) == len(config.patterns):
        out.findings.append(warning("chart", "chart_not_assessable", "The chart could not be judged: none of the numbers it needs could be made (too few stored sessions)."))
    elif out.score < config.weak_below:
        out.findings.append(warning("chart", "weak_chart", f"The chart is weak: {', '.join(h.name.replace('_', ' ') for h in hits if h.points < 0)} (score {out.score:g}, under {config.weak_below:g}).",
                                    value=out.score, threshold=config.weak_below))
    return out
