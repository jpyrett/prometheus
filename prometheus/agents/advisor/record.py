"""The shapes the Advisor writes: the recommendation record (schema 3), the candidate that did not make the list, and the parts they are made of (Advisor v1 spec section 14).

A record is the CONTRACT with the Critic: everything the Critic needs to judge a recommendation without asking the Advisor anything is in it. Each section answers one question:

    identity        which request, which case, which regime, when
    action          `open` (a recommendation to BUY: nothing else exists), and why
    families        which strategy families ran, whether they could be computed, what they scored, what they were scored FROM (`features_used`), the evidence ids
    quality         the haircut multiplier and every check that applied or was skipped
    evidence        the verified quotes (v0 shape), the filing-evidence items cited, the events, the counter-evidence, the contradictions
    features        every metric used, as the Data Clerk gave it: {name: {value, unit, as_of, source}}
    agreement       how much the text agrees, how complete the inputs are, and the confidence that results
    ranking         the composite edge, the cost, the expected excess return after cost, the stance, the rank
    trade           the plan: entry zone, stop, target, time stop, order style, invalidation rules (prices and dates: NO size, how much is the Account Manager's, entry 80)
    risk, concentration, gates, thesis (the model's words, checked), audit (which model calls, which config)

The v0 fields (`symbol, company, stance, model_score, score, supported, consistent, bullish_signals, bearish_signals, signals_claimed, signals_verified, signals_counted, signals, advisor`)
keep their meaning, so `account_manager.top_recommendation` and the week 3 lab read a v1 record unchanged. `score` is `round(50 + 50 * the blend)` (Amendment 6); the expected excess return after cost was removed from the record (Amendment 7), and so were the size of the trade and the user's money (entry 80: the Advisor says which companies and at what
prices, never how much).

THE ADVISOR IS GIVEN NO PORTFOLIO, so a record carries nothing about the user's holdings (design choices entry 105; Jacob, 2026-11-15: *"never any portfolio related information"*): every record is a recommendation to
buy, there is no `add`, `hold`, `reduce` or `avoid`, no number of shares held, no cost basis, no unrealized gain, no `held` flag on a candidate and no `portfolio_fit` (its two numbers were never computed; its note about how
this run's own picks sit together is `concentration.note`). A sale never comes from the Advisor: it comes from the user's own order or from the Account Manager's RaiseFunds.

SCHEMA 3 READS SCHEMA 2. A record made before that (the real captures of `tests/fixtures/session_real/` are version 2, every action `open`, nothing held) still loads: the retired keys are dropped when they hold the not-held
values (`held_shares` 0, `cost_basis` and `unrealized_pct` null, the two `portfolio_fit` numbers null, `held` false) and `portfolio_fit.note` is read as `concentration.note`. A record that is NOT a purchase (its action is not `open`) or that says
something is held is REFUSED, with the reason: it is not a recommendation the desk can use.

Every model here forbids unknown fields: a typo in the code that builds a record fails when the record is built, not when the Critic reads it. A test asserts that no field except the ones
listed in `MODEL_FIELDS` originates from a model's reply.
"""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 3   # 3 since design choices entry 105: no holding of the user's is on a record (action `open` only, no `held_shares`, `cost_basis`, `unrealized_pct`, `held` or `portfolio_fit`). It was kept at 2 through the amendments that took a field out (Amendment 7, entry 80): a field that goes is simply gone (the models forbid extras). Records stored before entry 80 still hold a size
# The ONLY places a model's words reach a record (spec 14): the model_score it gave, the thesis and bullets, the quotes of its signals and events, and the company name.
MODEL_FIELDS = ("model_score", "thesis", "bull_case", "bear_case", "signals[*].quote", "events[*].quote", "company")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Identity(_Model):
    model_config = ConfigDict(extra="ignore")      # a record made before design choices entry 97 holds a `market_cap` key that no longer exists; it is read and dropped
    schema_version: int = SCHEMA_VERSION
    record_id: str
    request_id: str
    case: Literal["funds", "theme", "industry", "market"]
    sleeve: str
    as_of_utc: str
    sector_etf: str | None = None
    regime: str
    regime_note: str | None = None


class Action(_Model):
    """A recommendation is to BUY, and nothing else (entry 105). A version 2 record's retired keys are dropped when they say nothing is held; any other action, or a holding, is refused."""
    action: Literal["open"]
    reason: str
    cooldown_of: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _read_version_2(cls, data):
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if data.get("action", "open") != "open":
            raise ValueError(f"the action is {data['action']!r}: a recommendation is to buy (`open`) and the Advisor records no holding, sale, hold or addition (a sale comes from the user's own order or RaiseFunds)")
        for key, empty in (("held_shares", 0), ("cost_basis", None), ("unrealized_pct", None)):
            if key in data:
                if data[key] not in (empty, None):
                    raise ValueError(f"the record says something is held ({key} {data[key]!r}): the Advisor knows no holding of the user's")
                del data[key]
        return data


class FamilyResult(_Model):
    family: str
    computable: bool                       # False: a REQUIRED input is missing (`missing` names them) and the family says nothing
    passed: bool = False                   # computable and every rule of the family held: only then is there a `score`
    missing: list[str] = Field(default_factory=list)
    waived: list[str] = Field(default_factory=list)   # rules that were NOT applied, and which: ["debt_to_equity"] for a company with negative stockholders' equity (design choices entry 54)
    scoring_basis: Literal["threshold", "rank"] = "threshold"
    score: float | None = None
    direction: Literal["bullish", "bearish"] | None = None
    horizon_bars: int
    prior_20d: float
    basis: Literal["prior", "measured"] = "prior"
    basis_ref: str = ""
    n: int = 0
    regime_multiplier: float = 1.0
    edge_20d: float | None = None
    completeness: float | None = None
    features_used: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    counter_evidence_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class Haircut(_Model):
    check: str
    value: Any
    factor: float


class Quality(_Model):
    multiplier: float
    haircuts: list[Haircut] = Field(default_factory=list)
    haircuts_skipped: list[str] = Field(default_factory=list)
    turned_off: list[str] = Field(default_factory=list)   # families a failing check switched off for this symbol


class Evidence(_Model):
    signals: list[dict] = Field(default_factory=list)      # the verified signals, v0 shape
    e_items: list[dict] = Field(default_factory=list)      # the filing-evidence items a verified signal cited, as the Data Clerk wrote them
    events: list[dict] = Field(default_factory=list)       # {class, direction, source, quote, origin, age_bars, reaction, reaction_atr}
    counter_evidence: list[dict] = Field(default_factory=list)   # verified signals and events against the direction of the stance
    contradictions: list[list[str]] = Field(default_factory=list)  # pairs of refs, a bullish and a bearish counted signal of the same company, derived by code


class Agreement(_Model):
    agreement: float
    evidence_n: int
    completeness: float | None
    confidence: float


class Ranking(_Model):
    composite: float | None = None
    conflict: bool = False
    stance: Literal["bullish", "bearish", "neutral"] = "neutral"
    stance_reason: str | None = None
    rank: int | None = None                # the place in the list this run returned
    universe_rank: int | None = None       # the place among ALL the new ideas evaluated, best to worst (a symbol that failed a gate is ranked after those that did not)
    diversity_adjustment: bool = False
    displaced_by: str | None = None
    concentrated: bool = False


class TradePlan(_Model):
    """The plan of one trade: prices and dates, no size (entry 80)."""
    family: str | None = None
    entry_basis: str
    entry_zone: list[float] | None = None
    entry_mid: float | None = None
    stop_distance: float | None = None
    stop_price: float | None = None
    target_price: float | None = None
    horizon_bars: int | None = None
    time_stop_date: str | None = None
    time_stop_note: str | None = None
    gap_risk: bool = False
    order_style: str | None = None
    limit_price: float | None = None
    invalidation: list[dict] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class Risk(_Model):
    model_config = ConfigDict(extra="ignore")      # a record made before design choices entry 93 holds two keys that no longer exist (the stock's daily dollars and its spread: the Advisor has no such input, and neither has the Data Clerk any more, entry 95); they are read and dropped
    vol_63d_ann: float | None = None
    atr_pct: float | None = None
    beta_63d: float | None = None
    max_drawdown_63d: float | None = None
    days_to_next_earnings: float | None = None
    earnings_move_abs: float | None = None
    gap_risk: bool = False
    red_flags: list[str] = Field(default_factory=list)


class Concentration(_Model):
    """How this run's OWN picks sit together ("three names are one bet"), worked out from the picks alone: nothing of the user's holdings is in it (entry 105; it was `portfolio_fit.note` in a version 2 record)."""
    note: str | None = None


class Gate(_Model):
    gate: str
    passed: bool
    value: Any = None
    threshold: Any = None
    note: str | None = None


class Thesis(_Model):
    thesis: str | None = None
    bull_case: list[dict] = Field(default_factory=list)   # [{text, ref}]
    bear_case: list[dict] = Field(default_factory=list)
    thesis_error: str | None = None


class Audit(_Model):
    metric_order: str | None = None
    plan_run: dict | None = None
    thesis_run: dict | None = None
    config_version: str
    family_versions: dict[str, str] = Field(default_factory=dict)


class Recommendation(_Model):
    # the version 0 fields, same meaning
    symbol: str
    company: str | None = None
    stance: str | None = None
    model_score: int | None = None
    score: int | None = None
    supported: bool = False
    consistent: bool | None = None
    bullish_signals: int = 0
    bearish_signals: int = 0
    signals_claimed: int = 0
    signals_verified: int = 0
    signals_counted: int = 0
    signals: list[dict] = Field(default_factory=list)
    advisor: dict = Field(default_factory=dict)
    # version 1
    blend: dict | None = None              # entry 55: the score's four parts (news 40%, performance 30%, historical 20%, filings 10%) with the pieces of each and their weights
    negative_equity: bool = False          # the company has a stockholders' deficit: its debt-to-equity rules were waived (see each family's `waived`, the red_flag gate's note and the quality `haircuts_skipped`)
    identity: Identity
    action: Action
    families: list[FamilyResult] = Field(default_factory=list)
    quality: Quality
    modifiers: list[dict] = Field(default_factory=list)   # the bench (insider, estimate revisions, short interest) is not built: always empty
    evidence: Evidence
    features: dict[str, dict] = Field(default_factory=dict)
    agreement: Agreement
    ranking: Ranking
    trade: TradePlan
    risk: Risk
    concentration: Concentration
    gates: list[Gate] = Field(default_factory=list)
    thesis: Thesis = Field(default_factory=Thesis)
    audit: Audit
    critic: dict | None = None             # the Critic's feedback (`critic/feedback.py`), put here by the Critic after its review: the Advisor never writes it

    @model_validator(mode="before")
    @classmethod
    def _read_version_2(cls, data):
        """A version 2 record's `portfolio_fit` becomes `concentration` (its two numbers were never computed: any other value than null is refused, and so is a record that is not new enough to be read)."""
        if not isinstance(data, dict) or "portfolio_fit" not in data:
            return data
        data = dict(data)
        fit = data.pop("portfolio_fit")
        if not isinstance(fit, dict) or any(fit.get(k) is not None for k in ("sector_weight_after", "correlation_to_portfolio_63d")) or set(fit) - {"sector_weight_after", "correlation_to_portfolio_63d", "note"}:
            raise ValueError("the record carries portfolio information (`portfolio_fit`): the Advisor knows no portfolio")
        data.setdefault("concentration", {"note": fit.get("note")})
        return data


class ExitPlan(_Model):
    """How a NEUTRAL company's position ends (Advisor v2, `design/critic_v2_evidence_spec.md` section 7): a stop under the last close and a time stop, NO target, nothing else. It is an EXIT plan only: the Trader's tickets, the entry
    zone, the limit range and the order type of a neutral purchase are what they were. `stop_price` is None when the ATR could not be calculated (the time stop alone stands); `basis` says how both were made."""
    stop_price: float | None = None
    time_stop_date: str
    basis: str


class Candidate(_Model):
    """A symbol that reached scoring and is not in the returned list, with the reason (a gate that stopped it, or its rank)."""
    symbol: str
    company: str | None = None
    origin: str
    gate: str | None = None
    gate_value: Any = None
    families: list[FamilyResult] = Field(default_factory=list)
    blend: dict | None = None              # the same blend as a record has
    negative_equity: bool = False          # a stockholders' deficit: the debt-to-equity rules were waived
    score: int | None = None               # the same 0 to 100 score a record has: the blend (None only when the company has no metrics)
    stance: str | None = None
    rank: int | None = None                # the place among all the new ideas evaluated, best to worst
    reason: str
    cooldown_of: str | None = None
    theme_role: str | None = None
    exit_plan: ExitPlan | None = None      # a NEUTRAL company (stance neutral, no failed gate) carries one; any other candidate has none; a candidate saved before Advisor config version 2 has none

    @model_validator(mode="before")
    @classmethod
    def _read_version_2(cls, data):
        """A version 2 candidate's `held: false` is dropped; `held: true` is refused (the Advisor knows no holding)."""
        if isinstance(data, dict) and "held" in data:
            data = dict(data)
            if data.pop("held"):
                raise ValueError("the candidate says it is held: the Advisor knows no holding of the user's")
        return data
