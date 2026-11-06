"""From family scores to a ranked short list: the regime, the expected excess return after cost, the stance, the holdings pass, the cooldown, the sleeves and the diversity rule
(Advisor v1 spec sections 9.5 to 9.8 and 12, design choices entry 51). Plain Python over numbers the families and the Data Clerk produced: nothing here calls a model.

The chain for one symbol:

    families (each a FamilyResult with an `edge_20d`)  ->  composite  ->  excess_20d_after_cost  ->  stance, score

    composite = the best bullish edge, minus the largest bearish magnitude when both exist (`conflict`); the only bullish edge; or the most negative edge when only bearish.
                One vote per FAMILY: two correlated trend features already became one trend score inside the family.
    cost_pct  = half the spread (0.10% when it is not known) + 0.10% slippage; doubled in the `stress` regime.
    confidence = (evidence_n x agreement + 4 x 0.5) / (evidence_n + 4): a prior of four neutral observations, so one quote is not certainty. Agreement is how much of the TEXT
                evidence points the way of the composite (a bearish composite uses 1 - agreement).
    excess_20d_after_cost = composite x (0.5 + confidence) - cost_pct          (confidence 0.5 multiplies by 1.0, confidence 1.0 by 1.5)
    stance    = bullish when that is above zero; bearish when the composite is below zero; else neutral ("edge_below_cost")
    score     = round(50 + 50 x clip(excess / 0.03, -1, 1)): the v0 field `top_recommendation` reads, so the Account Manager keeps working

The regime (computed once per run from SPY) scales the families through the config; the sleeve of the use case decides which families may OPEN a name and how many (`select`).
"""
from dataclasses import dataclass, field

from prometheus.agents.advisor.config import AdvisorConfig
from prometheus.agents.advisor.gates import GateResult, value
from prometheus.agents.advisor.record import FamilyResult, Quality

OPEN_FAMILIES_NEED_TREND_REGIME = {"funds": ("trend",)}   # in the funds sleeve a trend name may only OPEN while the regime is `trend`


@dataclass
class Scored:
    """Everything known about one symbol when it is ranked. The Advisor fills the first block, `rank_symbol` the second."""
    symbol: str
    company: str | None
    origin: str
    held: bool
    metrics: dict
    gates: GateResult
    quality: Quality
    families: list[FamilyResult]
    events: list[dict]
    signals: list[dict]
    votes: list[tuple[str, float]]                  # the text evidence: (direction, weight) per counted signal and event; a news item 1.0, a company's own words 0.5
    sector_etf: str | None = None
    held_shares: int = 0
    cost_basis: float | None = None
    theme_role: str | None = None
    # filled by rank_symbol
    composite: float | None = None
    conflict: bool = False
    agreement: float = 0.5
    evidence_n: int = 0
    confidence: float = 0.5
    blend: dict | None = None                       # `blend.compute_blend`: the 0.40 news / 0.30 performance / 0.20 historical / 0.10 filings score; None counts as 0 (neutral)
    scale_factor: float = 1.0                       # what multiplies the strategy's edge: 1 + scale_strength x direction x blend
    cost_pct: float | None = None
    excess: float | None = None
    stance: str = "neutral"
    stance_reason: str | None = None
    score: int | None = None
    notes: list[str] = field(default_factory=list)

    def passed(self, direction: str | None = None) -> list[FamilyResult]:
        return [f for f in self.families if f.passed and f.edge_20d is not None and (direction is None or f.direction == direction)]

    def vol(self) -> float:
        got = value(self.metrics, "vol_63d_ann")
        return got if got is not None else float("inf")  # an unknown volatility ranks after a known one on a tie


# ---------------------------------------------------------------- the regime

def regime_of(metrics: dict | None, config: AdvisorConfig) -> tuple[str, str | None]:
    """(state, note) from SPY's metrics (any symbol's `data.metrics` entry has them). `unknown` when SPY's bars are too few for its 200-day mean. The order of the tests: unknown,
    recovering (a 10% drawdown and a 5% bounce), stress (volatility over 25% or SPY under its 200-day mean), narrow (breadth under 40%), else trend."""
    rc = config.regime
    above = value(metrics, "spy_above_sma_200")
    if above is None:
        why = ((metrics or {}).get("spy_above_sma_200") or {}).get("note") or "SPY metrics are missing"
        return "unknown", f"{why}; regime not computable"
    drawdown, ret21, vol, breadth = value(metrics, "spy_drawdown_63d"), value(metrics, "spy_ret_21d"), value(metrics, "spy_vol_21d_ann"), value(metrics, "breadth_200")
    if drawdown is not None and ret21 is not None and drawdown < rc.recovering_drawdown and ret21 > rc.recovering_ret_21d:
        return "recovering", f"SPY fell {drawdown:.1%} over 63 bars and gained {ret21:.1%} in 21"
    if (vol is not None and vol > rc.stress_vol) or above is False:
        return "stress", "SPY is under its 200-day mean" if above is False else f"SPY volatility {vol:.1%} is over {rc.stress_vol:.0%}"
    if breadth is not None and breadth < rc.narrow_breadth:
        return "narrow", f"breadth {breadth:.0%} is under {rc.narrow_breadth:.0%}"
    return "trend", None if breadth is None else f"breadth {breadth:.0%}"


# ---------------------------------------------------------------- one symbol

def agreement_of(votes: list[tuple[str, float]]) -> tuple[float, int]:
    """(agreement, evidence_n): `(bullish weight + 1) / (bullish weight + bearish weight + 2)`; n is the number of counted signals and events."""
    bull = sum(w for d, w in votes if d == "bullish")
    bear = sum(w for d, w in votes if d == "bearish")
    return (bull + 1) / (bull + bear + 2), sum(1 for d, _ in votes if d in ("bullish", "bearish"))


def combine(families: list[FamilyResult]) -> tuple[float | None, bool]:
    """(composite, conflict) from the passed families' edges."""
    bulls = [f.edge_20d for f in families if f.passed and f.direction == "bullish" and f.edge_20d is not None]
    bears = [abs(f.edge_20d) for f in families if f.passed and f.direction == "bearish" and f.edge_20d is not None]
    if bulls and bears:
        return max(bulls) - max(bears), True
    if bulls:
        return max(bulls), False
    if bears:
        return -max(bears), False
    return None, False


def cost_of(metrics: dict | None, regime: str, config: AdvisorConfig) -> float:
    spread = value(metrics, "median_spread_bps_1d")
    half = spread / 10000 / 2 if spread is not None else config.ranking.default_half_spread
    cost = half + config.ranking.slippage
    return cost * config.ranking.stress_cost_multiplier if regime == "stress" else cost


def v1_score(excess: float | None, config: AdvisorConfig) -> int | None:
    """`round(50 + 50 x clip(excess / 0.03, -1, 1))`; None when there is no excess (the company has no metrics)."""
    if excess is None:
        return None
    return round(50 + 50 * max(-1.0, min(1.0, excess / config.ranking.edge_scale)))


def rank_symbol(s: Scored, regime: str, config: AdvisorConfig) -> Scored:
    """Fill the composite, agreement, confidence, cost, excess, stance and score of one symbol. Returns it."""
    s.composite, s.conflict = combine(s.families)
    s.agreement, s.evidence_n = agreement_of(s.votes)
    if s.composite is not None and s.composite < 0:
        s.agreement = 1 - s.agreement                      # a bearish composite is supported by the text that is bearish
    blended = (s.blend or {}).get("value", 0.0)       # entry 55: -1 to +1, 40% news and press releases (the newer the heavier), 30% performance, 20% historical, 10% filings
    direction = -1.0 if s.composite is not None and s.composite < 0 else 1.0
    s.scale_factor = 1 + config.blend.scale_strength * direction * blended
    s.confidence = (1 + direction * blended) / 2       # the blend on a 0 to 1 scale, on the side of the composite (scale_factor = 0.5 + confidence at the default strength)
    s.cost_pct = cost_of(s.metrics, regime, config)
    if s.composite is None:
        # No family passed: there is no edge to scale, so the expected excess is what the blend alone is worth, minus the cost: a company with good news, good prices and good numbers outranks one
        # without (entry 55). It is not a recommendation (the stance stays neutral: no strategy gave an entry). A company with no numbers at all (no metrics) has no score.
        s.excess = blended * config.blend.no_strategy_edge_scale - s.cost_pct if s.metrics is not None else None
        s.stance, s.stance_reason, s.score = "neutral", "no family passed", v1_score(s.excess, config)
        return s
    s.excess = s.composite * s.scale_factor - s.cost_pct
    if s.excess > 0:
        s.stance, s.stance_reason = "bullish", None
    elif s.composite < 0:
        s.stance, s.stance_reason = "bearish", None
    else:
        s.stance, s.stance_reason = "neutral", "edge_below_cost"
    s.score = v1_score(s.excess, config)
    return s


def sort_key(s: Scored):
    """Best first: the highest excess, then the higher confidence (more of the verified text evidence on the side of the composite; with no composite it is how bullish the text is), then fewer
    haircuts, then lower volatility, then the symbol (so the order is the same every run)."""
    return (-(s.excess if s.excess is not None else float("-inf")), -s.confidence, len(s.quality.haircuts), s.vol(), s.symbol)


# ---------------------------------------------------------------- invalidation and cooldown

def invalidation_fired(rules: list[dict], metrics: dict | None, events: list[dict]) -> list[str]:
    """Which of an earlier recommendation's invalidation rules have fired now: the price under its stop, a bearish class event, or the family's own metric condition. (`days_elapsed` is the time
    stop, which the cooldown reads as a date.) Returns the kinds, e.g. ["price_below"]."""
    fired = []
    last = value(metrics, "last_close")
    for rule in rules or []:
        kind, target = rule.get("kind"), rule.get("value")
        if kind == "price_below" and last is not None and target is not None and last < target:
            fired.append(kind)
        elif kind == "event_class" and any(e.get("class") in (target or []) for e in events):
            fired.append(kind)
        elif kind == "metric_crossed":
            if (target == "below_sma_200" and value(metrics, "above_sma_200") is False) or (target == "rsi_14_above_70" and (value(metrics, "rsi_14") or 0) > 70) \
                    or (target == "dilution_current" and value(metrics, "dilution_current") is True):
                fired.append(f"{kind}:{target}")
    return fired


def cooldown_of(s: Scored, family: str | None, records: list[dict], today_iso: str) -> str | None:
    """The record_id of an earlier recommendation that still stands: same symbol and family, its time stop not yet passed, and none of its invalidation rules fired. None when there is none."""
    for record in reversed(records):
        trade = record.get("trade") or {}
        if family and trade.get("family") != family:
            continue
        if (record.get("action") or {}).get("action") not in ("open", "add"):
            continue
        if (trade.get("time_stop_date") or "") >= today_iso and not invalidation_fired(trade.get("invalidation") or [], s.metrics, s.events):
            return (record.get("identity") or {}).get("record_id")
    return None


# ---------------------------------------------------------------- holdings

def holding_action(s: Scored, records: list[dict], plan_oversized: bool, position_pct_after: float | None, add_floor: float | None, config: AdvisorConfig) -> tuple[str, str]:
    """(action, reason) for a symbol the portfolio holds: the first rule that matches.
    reduce  a red_flag gate failed; or a bearish class event of the last 10 bars; or an earlier recommendation's invalidation rule fired; or the family that recommended it before
            no longer passes (None or bearish);
    add     the stance is bullish, it is not oversized, its excess would have made this run's list (`add_floor`: the lowest excess among the new ideas picked; None when there are
            none), and the position after adding stays within the limit;
    hold    everything else."""
    gate = next((g for g in s.gates.gates if g.gate == "red_flag" and not g.passed), None)
    if gate:
        return "reduce", f"red_flag gate failed: {gate.note}"
    bearish = next((e for e in s.events if e.get("direction") == "bearish" and e.get("age_bars") is not None and e["age_bars"] <= 10), None)
    if bearish:
        return "reduce", f"a bearish event in the last 10 bars: {bearish['class']}"
    for record in reversed(records):
        trade = record.get("trade") or {}
        if (record.get("action") or {}).get("action") not in ("open", "add"):
            continue
        fired = invalidation_fired(trade.get("invalidation") or [], s.metrics, s.events)
        if fired:
            return "reduce", f"an earlier recommendation ({(record.get('identity') or {}).get('record_id')}) was invalidated: {', '.join(fired)}"
        original = next((f for f in s.families if f.family == trade.get("family")), None)
        if original is not None and original.computable and (not original.passed or original.direction == "bearish"):
            return "reduce", f"the family that recommended it ({original.family}) no longer passes"
        break
    if s.stance == "bullish" and not plan_oversized and (add_floor is None or (s.excess or 0) >= add_floor) and position_pct_after is not None and position_pct_after <= config.ranking.add_max_position_pct:
        return "add", f"bullish, excess {s.excess:.2%}, position after adding {position_pct_after:.0%}"
    return "hold", "no reason to change the position" if s.stance != "bullish" else "bullish, but not enough to add (oversized, limit, or below this run's new ideas)"


# ---------------------------------------------------------------- the sleeve

def open_family(s: Scored, case: str, regime: str, config: AdvisorConfig) -> FamilyResult | None:
    """The family a name would be OPENED on: the passed bullish family with the highest edge among those the case's sleeve allows (and not blocked by a gate). In the funds sleeve `trend`
    only counts in the `trend` regime. None when no allowed family qualifies."""
    sleeve = config.sleeves[case]
    allowed = [f for f in s.passed("bullish") if f.family in sleeve.eligible and not s.gates.blocks(f.family)
               and not (f.family in OPEN_FAMILIES_NEED_TREND_REGIME.get(case, ()) and regime != "trend")]
    return max(allowed, key=lambda f: f.edge_20d) if allowed else None


def select(candidates: list[Scored], case: str, regime: str, config: AdvisorConfig) -> tuple[list[tuple[Scored, FamilyResult, dict]], dict[str, str]]:
    """Choose the new ideas to OPEN. Returns ([(scored, the family it opens on, {diversity_adjustment, displaced_by})] in rank order, {symbol: why it was not chosen}).
    A name must: pass its gates for an allowed family, have a bullish stance and a positive composite, not be vetoed (the funds sleeve lets `event` only veto: a bearish event vote blocks
    an open), and, in the theme sleeve, pass the `theme` family. In the funds sleeve a name that only an `event` family supports is used ONLY when no other name qualifies.
    Order: the highest excess, fewer haircuts, lower volatility, symbol. Diversity: when the next pick is in the same sector ETF as one already picked, the best name of another sector
    (or of an unknown one) whose excess is within 80% of it is taken instead, and both facts are recorded."""
    sleeve, why = config.sleeves[case], {}
    primary, fallback = [], []
    for s in candidates:
        family = open_family(s, case, regime, config)
        vetoes = [f.family for f in s.passed("bearish") if f.family in sleeve.veto_only]
        if s.stance != "bullish" or (s.composite or 0) <= 0:
            why[s.symbol] = f"stance {s.stance}" + (f" ({s.stance_reason})" if s.stance_reason else "")
        elif vetoes:
            why[s.symbol] = f"vetoed by a bearish {vetoes[0]} vote"
        elif case == "theme" and not any(f.family == "theme" for f in s.passed("bullish")):
            why[s.symbol] = "the theme family did not pass (theme_role: " + (s.theme_role or "none") + ")"
        elif family is not None:
            primary.append((s, family))
        elif case == "funds" and any(f.family == "event" for f in s.passed("bullish")) and not s.gates.blocks("event"):
            fallback.append((s, next(f for f in s.passed("bullish") if f.family == "event")))
        else:
            blocked = s.gates.first_failed()
            why[s.symbol] = f"gate {blocked.gate} failed" if blocked else "no family the sleeve allows passed"
    pool = sorted(primary or fallback, key=lambda pair: sort_key(pair[0]))
    for s, _ in (fallback if primary else []):
        why[s.symbol] = "only an event family supports it, and other names qualify"
    picked: list[tuple[Scored, FamilyResult, dict]] = []
    remaining = list(pool)
    while remaining and len(picked) < sleeve.top_k:
        first = remaining[0]
        choice, extra = first, {"diversity_adjustment": False, "displaced_by": None}
        taken = {p[0].sector_etf for p in picked if p[0].sector_etf}
        if first[0].sector_etf and first[0].sector_etf in taken:
            alt = next((p for p in remaining[1:] if p[0].sector_etf != first[0].sector_etf and (p[0].excess or 0) >= config.ranking.diversity_within * (first[0].excess or 0)), None)
            if alt is not None:
                choice, extra = alt, {"diversity_adjustment": True, "displaced_by": first[0].symbol}
        remaining.remove(choice)
        picked.append((choice[0], choice[1], extra))
    chosen = {p[0].symbol for p in picked}
    for position, (s, _) in enumerate(pool, 1):
        if s.symbol not in chosen:
            why[s.symbol] = f"rank {position} is beyond the top {sleeve.top_k}"
    return picked, why
