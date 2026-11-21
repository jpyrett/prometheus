"""The statistics of a run, each with its `n` and a small-sample mark; PURE (design/proving_ground_spec.md section 8.3, P9 and P12).

    stats = compute(outcomes, equity, config, starting_cash=Decimal("10000.00"))        # a `record.Stats`
    stats.headline.sentence            # "Beat SPY by 3.2 points over 24 weeks: 31 recommendations, 58% of the closed ones won."
    stats.critic_value.sentence        # "The Critic was right: the 9 it denied averaged -2.1 points against SPY, and the 22 it approved averaged +1.3."

P12: every number on the page is computed here, once, and the page draws. A program computes, no model takes part (entries 24 and 26). The sentences are code's own and come from a CLOSED set of templates (`HEADLINES`, `CRITIC_SENTENCES`,
the clauses `CLOSED_CLAUSES`): the numbers are put into a template, nothing else is ever written.

WHAT IS COUNTED. `outcomes` is every `RecOutcome` of the run, traded or hypothetical (P8: the denied ones too). An outcome is SCORED when it won, lost or is open (it has a return) and USABLE when it also has an excess return over the benchmark.
 - `portfolio`: from the equity curve alone: the return against the starting cash (`starting_cash`, else the first point's equity), the benchmark's, the excess, the worst drawdown (against the highest equity so far, the starting cash counting; the
   day of that peak and of the bottom), the best and the worst day, the days, and the average share of the account that was invested. None without a curve (the Track Record has none).
 - `outcomes`: every outcome: `n` = closed + open + not_filled + no_data; `win_rate` = won / closed (an open position is NEVER a win; None with nothing closed); the average win and the average loss are the mean returns of the winners and of the
   losers; the profit factor is the sum of the winning returns over the sum of the losing ones (None without a loss); `avg_sessions_held` is over the closed; `exits` counts how the closed ones ended and the open ones; `small_sample` is
   `closed < min_sample`.
 - `by_verdict` (approved, denied, neutral, unreviewed: the Critic's value), `by_score` (the Advisor's score in the buckets of the config: the calibration) and `by_family`: each group's `n` is its USABLE outcomes (what its means rest on),
   `closed` those of them that closed (what the win rate rests on), and `small_sample` is `n < min_sample`: under it the page shows a rate or an average with the mark and no colour of success or failure (P9). A bucket is [low, high).
 - `by_symbol`: the TRADED outcomes of each symbol (hypothetical ones have no money): their count, their gain (a sum of Money) and the gain over what was paid.

A `_pct` is a FRACTION (0.0077 is 0.77%), to six places, half up, worked out in Decimal from the digits the returns are written with so that it carries no binary noise (`simulator._fraction` and `_return_difference` are the one place a return
is worked out); a mean is the sum of the Decimals over the count. A return, a drawdown, a rate and a mean are measurements (P7): the only floats here.
"""
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

from prometheus.agents.trader import simulator
from prometheus.proving.config import ProvingConfig, load_config
from prometheus.proving.record import (CriticValue, EquityPoint, Exits, FamilyStats, Headline, OutcomeStats, PortfolioStats, RecOutcome, ScoreStats, Stats, SymbolStats, VerdictStats)

Measure = float | None      # a return, a drawdown, a rate or a mean: a measurement, never added to money (P7)
STEP = Decimal("0.000001")
VERDICTS = ("approved", "denied", "neutral", "unreviewed")

# The closed set of sentences (P12). The numbers are the only thing put into them.
CLOSED_CLAUSES = {"rate": "{rate} of the closed ones won", "none": "none has closed yet"}
HEADLINES = {
    "none": "No recommendation has an outcome yet.",
    "beat": "Beat {bench} by {points} points over {span}: {n} recommendations, {closed_clause}.",
    "trail": "Trailed {bench} by {points} points over {span}: {n} recommendations, {closed_clause}.",
    "match": "Matched {bench} over {span}: {n} recommendations, {closed_clause}.",
    "return": "Returned {ret} over {span}: {n} recommendations, {closed_clause}.",
    "small": "Small sample: {closed} closed so far, {min_sample} needed. The desk returned {ret} against {bench}'s {bret} over {span}; {n} recommendations are too few to call it.",
    "small_alone": "Small sample: {closed} closed so far, {min_sample} needed. The desk returned {ret} over {span}; {n} recommendations are too few to call it.",
    "avg_beat": "On average the recommendations beat {bench} by {points} points: {n} recommendations, {closed_clause}.",
    "avg_trail": "On average the recommendations trailed {bench} by {points} points: {n} recommendations, {closed_clause}.",
    "avg_match": "On average the recommendations matched {bench}: {n} recommendations, {closed_clause}.",
    "avg_small": "Small sample: {closed} closed so far, {min_sample} needed. On average the recommendations were {points} points against {bench}; {n} recommendations are too few to call it.",
    "avg_unknown": "{n} recommendations, {closed_clause}: there is no benchmark return to compare them with.",
}
CRITIC_SENTENCES = {
    "none": "The Critic has no outcomes to judge yet.",
    "no_denials": "The Critic denied nothing here, so there is no denial to judge. The {approved_n} it approved averaged {approved_mean} points against {bench}.",
    "no_approvals": "The Critic approved nothing here. The {denied_n} it denied averaged {denied_mean} points against {bench}.",
    "small": "Small sample: the Critic denied {denied_n} and approved {approved_n}, too few to say whether it was right ({min_sample} of each are needed).",
    "right": "The Critic was right: the {denied_n} it denied averaged {denied_mean} points against {bench}, and the {approved_n} it approved averaged {approved_mean}.",
    "wrong": "The Critic was wrong: the {denied_n} it denied averaged {denied_mean} points against {bench}, better than the {approved_n} it approved ({approved_mean}).",
    "unclear": "No clear difference: the {denied_n} the Critic denied averaged {denied_mean} points against {bench} and the {approved_n} it approved {approved_mean}; a gap of this size is what chance alone gives with this many.",
    "even": "The Critic made no difference: the {denied_n} it denied and the {approved_n} it approved averaged the same, {approved_mean} points against {bench}.",
}


def _q(value: Decimal) -> Measure:
    """A measurement as the float that is written: six places, half up."""
    return float(value.quantize(STEP, rounding=ROUND_HALF_UP))


def _mean(values) -> Measure:
    """The mean of returns (floats written to six places): the sum of the Decimals of their digits over the count, to six places; None for none."""
    values = list(values)
    if not values:
        return None
    return _q(sum((Decimal(str(v)) for v in values), Decimal(0)) / len(values))


def _rate(part: int, whole: int) -> Measure:
    return None if whole == 0 else _q(Decimal(part) / Decimal(whole))


def _scored(outcome: RecOutcome) -> bool:
    return outcome.result in ("won", "lost", "open") and outcome.return_pct is not None


def _usable(outcome: RecOutcome) -> bool:
    return _scored(outcome) and outcome.excess_return_pct is not None


def _group(rows: list[RecOutcome], config: ProvingConfig) -> dict:
    """The numbers every group of outcomes (a verdict, a score bucket, a family) carries."""
    closed = [r for r in rows if r.result in ("won", "lost")]
    return {"n": len(rows), "closed": len(closed), "mean_return_pct": _mean(r.return_pct for r in rows), "mean_excess_pct": _mean(r.excess_return_pct for r in rows),
            "win_rate": _rate(sum(1 for r in closed if r.result == "won"), len(closed)), "small_sample": len(rows) < config.min_sample}


# ---------------------------------------------------------------- the account

def _portfolio(equity: list[EquityPoint], starting_cash: Decimal | None) -> PortfolioStats | None:
    if not equity:
        return None
    points = sorted(equity, key=lambda p: p.date)
    start = starting_cash if starting_cash is not None else points[0].equity
    if start <= 0:
        return None
    end = points[-1].equity
    bench_last = next((p.benchmark for p in reversed(points) if p.benchmark is not None), None)
    returned = simulator._fraction(end, start)
    bench = None if bench_last is None else simulator._fraction(bench_last, start)
    peak, peak_day, worst, worst_from, worst_to = start, points[0].date, Decimal(0), None, None
    days, previous = [], start
    for point in points:
        if point.equity > peak:
            peak, peak_day = point.equity, point.date
        depth = point.equity / peak - 1
        if depth < worst:
            worst, worst_from, worst_to = depth, peak_day, point.date
        days.append(simulator._fraction(point.equity, previous))
        previous = point.equity
    invested = [p.invested / p.equity for p in points if p.equity > 0]
    return PortfolioStats(start=points[0].date, end=points[-1].date, start_equity=start, end_equity=end, return_pct=returned, benchmark_return_pct=bench,
                          excess_return_pct=None if bench is None else simulator._return_difference(returned, bench), max_drawdown_pct=_q(worst), drawdown_from=worst_from, drawdown_to=worst_to,
                          best_day_pct=max(days), worst_day_pct=min(days), days=len(points), invested_avg_pct=_q(sum(invested, Decimal(0)) / len(invested)) if invested else None)


# ---------------------------------------------------------------- the outcomes

def _outcome_block(outcomes: list[RecOutcome], config: ProvingConfig) -> OutcomeStats:
    won = [o for o in outcomes if o.result == "won"]
    lost = [o for o in outcomes if o.result == "lost"]
    closed = won + lost
    wins, losses = sum((Decimal(str(o.return_pct)) for o in won), Decimal(0)), sum((Decimal(str(o.return_pct)) for o in lost), Decimal(0))
    exits = Exits(open=sum(1 for o in outcomes if o.result == "open"), **{reason: sum(1 for o in closed if o.exit.reason == reason) for reason in ("stop", "target", "time_stop")})
    return OutcomeStats(n=len(outcomes), closed=len(closed), open=exits.open, won=len(won), lost=len(lost), not_filled=sum(1 for o in outcomes if o.result == "not_filled"), no_data=sum(1 for o in outcomes if o.result == "no_data"),
                        win_rate=_rate(len(won), len(closed)), avg_win_pct=_mean(o.return_pct for o in won), avg_loss_pct=_mean(o.return_pct for o in lost),
                        profit_factor=_q(wins / -losses) if lost and losses < 0 else None, avg_sessions_held=_mean(o.exit.sessions_held for o in closed), exits=exits, small_sample=len(closed) < config.min_sample)


def _by_symbol(outcomes: list[RecOutcome]) -> list[SymbolStats]:
    rows = []
    for symbol in sorted({o.symbol for o in outcomes if o.traded and _scored(o)}):
        mine = [o for o in outcomes if o.traded and _scored(o) and o.symbol == symbol]
        gain, paid = sum((o.gain for o in mine), Decimal("0.00")), sum((o.entry.amount for o in mine), Decimal("0.00"))
        rows.append(SymbolStats(symbol=symbol, n=len(mine), gain=gain, return_pct=_q(gain / paid) if paid > 0 else None))
    return rows


# ---------------------------------------------------------------- the sentences

def _points(value: Measure, sign: bool = False) -> str:
    """A fraction as points (percentage points), one decimal: 0.032 is '3.2' ('+3.2' with `sign`)."""
    points = (Decimal(str(value)) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return format(points, "+f" if sign else "f")


def _percent(value: Measure) -> str:
    return format((Decimal(str(value)) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP), "+f") + "%"


def _span(first: date, last: date) -> str:
    days = (last - first).days
    if days < 14:
        return f"{days} day{'' if days == 1 else 's'}"
    return f"{(days + 3) // 7} weeks"


def _clause(block: OutcomeStats) -> str:
    if block.closed == 0:
        return CLOSED_CLAUSES["none"]
    return CLOSED_CLAUSES["rate"].format(rate=f"{int((Decimal(str(block.win_rate)) * 100).quantize(Decimal(1), rounding=ROUND_HALF_UP))}%")


def _headline(portfolio: PortfolioStats | None, block: OutcomeStats, usable: list[RecOutcome], config: ProvingConfig, bench: str) -> Headline:
    if block.n == 0 or (block.closed + block.open == 0 and portfolio is None):
        return Headline(tone="neutral", sentence=HEADLINES["none"])
    said = {"n": block.n, "closed": block.closed, "min_sample": config.min_sample, "bench": bench, "closed_clause": _clause(block)}
    if portfolio is not None:
        said.update(ret=_percent(portfolio.return_pct), span=_span(portfolio.start, portfolio.end))
        excess = portfolio.excess_return_pct
        if excess is None:
            kind, tone = ("small_alone" if block.small_sample else "return"), ("neutral" if block.small_sample or portfolio.return_pct == 0 else "pos" if portfolio.return_pct > 0 else "neg")
        else:
            said.update(points=_points(abs(excess)), bret=_percent(portfolio.benchmark_return_pct))
            even = _points(abs(excess)) == "0.0"
            kind = "small" if block.small_sample else "match" if even else "beat" if excess > 0 else "trail"
            tone = "neutral" if block.small_sample or even else "pos" if excess > 0 else "neg"
        return Headline(tone=tone, sentence=HEADLINES[kind].format(**said))
    mean = _mean(o.excess_return_pct for o in usable)
    if mean is None:
        return Headline(tone="neutral", sentence=HEADLINES["avg_unknown"].format(**said))
    said["points"] = _points(abs(mean)) if not block.small_sample else _points(mean, sign=True)
    even = _points(abs(mean)) == "0.0"
    kind = "avg_small" if block.small_sample else "avg_match" if even else "avg_beat" if mean > 0 else "avg_trail"
    return Headline(tone="neutral" if block.small_sample or even else "pos" if mean > 0 else "neg", sentence=HEADLINES[kind].format(**said))


CHANCE_BAND = Decimal(2)       # a difference of two means is called only when it is more than this many standard errors of the difference (about 19 in 20 by chance alone would be inside it)


def spread(values: list[Decimal]) -> tuple[Decimal, Decimal]:
    """(the mean, the variance of the mean) of some measurements: the sample variance (n - 1) over n. Zero variance for fewer than two."""
    n = len(values)
    mean = sum(values, Decimal(0)) / n
    if n < 2:
        return mean, Decimal(0)
    return mean, sum(((v - mean) ** 2 for v in values), Decimal(0)) / (n - 1) / n


_spread = spread      # the name the module used before the report card (`report_card.py`) needed the same arithmetic


def gap_of(first: list, second: list) -> dict | None:
    """THE ONE PLACE a gap between two groups of excess returns is judged (the Critic's sentence, `stats.compute`, and the report card of the Critic's rules, `report_card.py`, both use it): {n_first, n_second, mean_first, mean_second, gap
    (the first's mean minus the second's), se (the standard error of that gap: the square root of the two variances of the means added), inside_chance (the gap is within `CHANCE_BAND` standard errors of zero: chance alone gives a
    gap this size with this many)}. Worked in Decimal from the digits the returns are written with; the numbers it gives are measurements (floats, to six places). None when either group is empty."""
    if not first or not second:
        return None
    (mean_a, var_a), (mean_b, var_b) = (spread([Decimal(str(v)) for v in group]) for group in (first, second))
    error = (var_a + var_b).sqrt()
    return {"n_first": len(first), "n_second": len(second), "mean_first": _q(mean_a), "mean_second": _q(mean_b), "gap": _q(mean_a - mean_b), "se": _q(error), "inside_chance": abs(mean_a - mean_b) <= CHANCE_BAND * error}


def _inside_chance(denied: list[RecOutcome], approved: list[RecOutcome]) -> bool:
    """True when the gap between the two groups' mean excess returns is within `CHANCE_BAND` standard errors of that gap: with this many outcomes, and returns this scattered, chance alone gives a gap of this size, so the
    sentence must not say the Critic was right or wrong (P9: honest about the sample). Worked by `gap_of`, from the outcomes' own excess returns."""
    return gap_of([o.excess_return_pct for o in denied], [o.excess_return_pct for o in approved])["inside_chance"]


def _critic_value(by_verdict: list[VerdictStats], config: ProvingConfig, bench: str, usable: list[RecOutcome] | None = None) -> CriticValue:
    denied, approved = next(v for v in by_verdict if v.verdict == "denied"), next(v for v in by_verdict if v.verdict == "approved")
    said = {"denied_n": denied.n, "approved_n": approved.n, "min_sample": config.min_sample, "bench": bench}
    for name, group in (("denied", denied), ("approved", approved)):
        said[f"{name}_mean"] = "n/a" if group.mean_excess_pct is None else _points(group.mean_excess_pct, sign=True)
    if denied.n == 0 and approved.n == 0:
        kind = "none"
    elif denied.n == 0:
        kind = "no_denials"
    elif approved.n == 0:
        kind = "no_approvals"
    elif denied.small_sample or approved.small_sample:
        kind = "small"
    elif usable is not None and _inside_chance([o for o in usable if o.verdict == "denied"], [o for o in usable if o.verdict == "approved"]):
        kind = "unclear"
    elif denied.mean_excess_pct < approved.mean_excess_pct:
        kind = "right"
    elif denied.mean_excess_pct > approved.mean_excess_pct:
        kind = "wrong"
    else:
        kind = "even"
    return CriticValue(denied_n=denied.n, denied_mean_excess_pct=denied.mean_excess_pct, approved_n=approved.n, approved_mean_excess_pct=approved.mean_excess_pct, sentence=CRITIC_SENTENCES[kind].format(**said))


def compute(outcomes: list[RecOutcome], equity: list[EquityPoint], config: ProvingConfig | None = None, *, starting_cash: Decimal | None = None, benchmark: str | None = None) -> Stats:
    """The statistics of a run: `outcomes` (every recommendation's, traded or hypothetical), `equity` (the curve, oldest first; empty for a run with none), `starting_cash` (the base of the account's return; the first point's equity when left
    out), `benchmark` (the name the sentences use; the config's). See the module text for every number."""
    config = config or load_config()
    bench = benchmark or config.benchmark
    usable = [o for o in outcomes if _usable(o)]
    portfolio, block = _portfolio(equity, starting_cash), _outcome_block(outcomes, config)
    by_verdict = [VerdictStats(verdict=v, **_group([o for o in usable if o.verdict == v], config)) for v in VERDICTS]
    by_score = [ScoreStats(low=low, high=high, **_group([o for o in usable if o.advisor and config.bucket_of(o.advisor.score) == (low, high)], config)) for low, high in config.buckets()]
    by_family = [FamilyStats(family=f, **_group([o for o in usable if o.advisor and o.advisor.family == f], config)) for f in sorted({o.advisor.family for o in usable if o.advisor and o.advisor.family})]
    return Stats(portfolio=portfolio, outcomes=block, by_verdict=by_verdict, by_score=by_score, by_family=by_family, by_symbol=_by_symbol(outcomes), critic_value=_critic_value(by_verdict, config, bench, usable),
                 headline=_headline(portfolio, block, usable, config, bench))
