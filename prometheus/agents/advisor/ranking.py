"""From family scores to a ranked short list: the regime, the stance, the blend that ranks, the holdings pass, the cooldown, the sleeves and the diversity rule
(Advisor v1 spec sections 9.5 to 9.8 and 12, design choices entry 51). Plain Python over numbers the families and the Data Clerk produced: nothing here calls a model.

The chain for one symbol:

    families (each a FamilyResult with an `edge_20d`)  ->  composite  ->  stance, score

    composite = the best bullish edge, minus the largest bearish magnitude when both exist (`conflict`); the only bullish edge; or the most negative edge when only bearish.
                One vote per FAMILY: two correlated trend features already became one trend score inside the family.
    confidence = (evidence_n x agreement + 4 x 0.5) / (evidence_n + 4): a prior of four neutral observations, so one quote is not certainty. Agreement is how much of the TEXT
                evidence points the way of the composite (a bearish composite uses 1 - agreement).
    (There is no expected return and no cost in the chain: Amendment 7 removed the expected 20-day excess return after cost, which was a product of unmeasured priors.)
    stance    = bullish when a bullish strategy passed (the composite is above zero); bearish when the composite is below zero; else neutral
    score     = round(50 + 50 x the blend): the v0 field `top_recommendation` reads, so the Account Manager keeps working. The blend (news and press releases 40%, performance 30%, history 20%,
                filings 10%) is what ranks the candidates, highest first (`rank_value`).

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


def rank_value(s: Scored) -> float:
    """What ranks a symbol (Amendment 6): its blend, -1 to +1 (news, performance, history and filings together). A symbol with no blend ranks as 0."""
    return (s.blend or {}).get("value", 0.0)


def blend_score(s: Scored) -> int | None:
    """`round(50 + 50 x the blend)`; None when the company has no metrics at all (nothing was calculated for it)."""
    return None if s.metrics is None else round(50 + 50 * rank_value(s))


def rank_symbol(s: Scored) -> Scored:
    """Fill the composite, agreement, confidence, stance and score of one symbol. Returns it."""
    s.composite, s.conflict = combine(s.families)
    s.agreement, s.evidence_n = agreement_of(s.votes)
    if s.composite is not None and s.composite < 0:
        s.agreement = 1 - s.agreement                      # a bearish composite is supported by the text that is bearish
    blended = rank_value(s)                            # -1 to +1: 40% news and press releases (the newer the heavier), 30% performance, 20% historical, 10% filings
    direction = -1.0 if s.composite is not None and s.composite < 0 else 1.0
    s.confidence = (1 + direction * blended) / 2       # the blend on a 0 to 1 scale, on the side of the composite
    s.score = blend_score(s)
    if s.composite is None:
        s.stance, s.stance_reason = "neutral", "no family passed"      # not a recommendation: no strategy gave an entry
    elif s.composite > 0:
        s.stance, s.stance_reason = "bullish", None
    elif s.composite < 0:
        s.stance, s.stance_reason = "bearish", None
    else:
        s.stance, s.stance_reason = "neutral", "no edge"
    return s


def sort_key(s: Scored):
    """Best first: the highest blend, then the larger strategy edge, then the higher confidence (more of the verified text evidence on the side of the composite), then fewer haircuts, then
    lower volatility, then the symbol (so the order is the same every run)."""
    return (-(rank_value(s) if s.metrics is not None else float("-inf")), -(s.composite or 0.0), -s.confidence, len(s.quality.haircuts), s.vol(), s.symbol)


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
            if (target == "below_sma_200" and value(metrics, "above_sma_200") is False) or (target == "rsi_14_above_70" and (value(metrics, "rsi_14") or 0) > 70):
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

def holding_action(s: Scored, records: list[dict], add_floor: float | None) -> tuple[str, str]:
    """(action, reason) for a symbol the portfolio holds: the first rule that matches.
    reduce  a red_flag gate failed; or a bearish class event of the last 10 bars; or an earlier recommendation's invalidation rule fired; or the family that recommended it before
            no longer passes (None or bearish);
    add     the stance is bullish and its blend would have made this run's list (`add_floor`: the lowest blend among the new ideas picked; None when there are none);
    hold    everything else.
    THE ADVISOR KNOWS NO POSITION SIZE (design choices entry 80). This rule used to ask also that one share was not too big for the account and that the position after adding stayed within 20% of the
    user's total money: both needed a size and the user's money. They are gone. Whether the user's own limits leave room to add is the Account Manager's: `sizing.size_purchases` counts the shares already held
    against the position limit, and an `add` that no limit leaves room for becomes `no_funds` with the reason, never a trade. A `reduce` carries no amount either: the Account Manager decides how
    much to sell."""
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
    if s.stance == "bullish" and (add_floor is None or rank_value(s) >= add_floor):
        return "add", f"bullish, blended score {s.score}, as good as this run's new ideas"
    return "hold", "no reason to change the position" if s.stance != "bullish" else "bullish, but below this run's new ideas"


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
    Order: the highest blend, the larger edge, fewer haircuts, lower volatility, symbol. Diversity: when the next pick is in the same sector ETF as one already picked, the best name of another sector
    (or of an unknown one) whose blend is within 80% of it is taken instead, and both facts are recorded."""
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
            alt = next((p for p in remaining[1:] if p[0].sector_etf != first[0].sector_etf and rank_value(p[0]) >= config.ranking.diversity_within * rank_value(first[0])), None)
            if alt is not None:
                choice, extra = alt, {"diversity_adjustment": True, "displaced_by": first[0].symbol}
        remaining.remove(choice)
        picked.append((choice[0], choice[1], extra))
    chosen = {p[0].symbol for p in picked}
    for position, (s, _) in enumerate(pool, 1):
        if s.symbol not in chosen:
            why[s.symbol] = f"rank {position} is beyond the top {sleeve.top_k}"
    return picked, why
