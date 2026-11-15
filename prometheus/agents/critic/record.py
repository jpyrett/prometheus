"""The shapes the Critic writes: one `Review` per recommendation, and the `Finding`s it is made of (Critic v1 spec sections 8 to 10).

A review answers the questions a person asks of a recommendation, in the order they ask them:

    approved, score, grade   the verdict, and the score it comes from (0 to 100; a blocker caps it)
    blockers, warnings       what is wrong (a blocker denies), what to know (a warning does not), notes (information)
    dimensions               the six parts of the score with their weights: accuracy, evidence, risk fit, timing, value, chart
    accuracy                 what was fetched again and whether it matched; the sources read again; the cross-checks
    evidence                 what the model said about each quote (labels only), the control claim, newer sources, what the expected return is WITHOUT the quotes that did not hold
    timing                   the session, the record's date, the price and where it sits in the entry zone
    risk                     the tolerance, the risk score and level (never a size: how much to buy is the Account Manager's)
    value                    the price against the six-month average, the profit potential, the bonus
    chart                    the chart patterns that hold (golden cross, dead cat bounce, ...) and the numbers behind them
    audit                    which model, which config, which Data Clerk orders

Every model forbids unknown fields: a typo in the code that builds a review fails when the review is built. The reviews live on `order.reviews` as dicts (`model_dump(mode="json")`); the
Advisor's records are never touched (spec P5). The ONLY places a model's words reach a review are listed in `MODEL_FIELDS`, and a test asserts it.
"""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 1      # kept at 1 through the amendments (entries 62, 65), as the Critic's feedback block is: a field that goes is simply gone (the models forbid extras). A review carries NO SIZE any more (sizing is the Account Manager's); reviews stored before that still hold one
SEVERITIES = ("blocker", "warning", "info")
# The ONLY places a model's words reach a review (spec 10): the labels and reasons of the claims, the control's label, and the label and quote of a fresh source.
MODEL_FIELDS = ("evidence.claims[*].label", "evidence.claims[*].reason", "evidence.control.label", "evidence.fresh[*].label", "evidence.fresh[*].quote")

# The closed list of codes that may be a BLOCKER (spec 9.3), in the order the summary names them. A blocker with any other code cannot be built.
BLOCKER_CODES: dict[str, tuple[str, ...]] = {
    "accuracy": ("record_not_reviewable", "record_defect", "data_unverified", "data_mismatch", "source_missing", "source_altered", "symbol_unknown", "price_mismatch"),
    "timing": ("market_closed", "record_expired", "time_stop_passed", "bars_stale", "no_live_price", "quote_stale", "price_left_zone", "stop_already_hit"),
    "risk_fit": ("risk_above_profile", "symbol_excluded", "sector_excluded", "price_below_profile_min"),
    "value": ("overpriced", "valuation_extreme", "falling_knife"),
    "chart": ("dead_cat_bounce",),
    "evidence": ("event_not_supported", "blend_not_supported", "fresh_negative_event"),
}
BLOCKER_ORDER = tuple(code for codes in BLOCKER_CODES.values() for code in codes)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Finding(_Model):
    """One thing the Critic found: a plain sentence with its numbers in it."""
    code: str
    dimension: Literal["accuracy", "evidence", "risk_fit", "timing", "value", "chart"]
    severity: Literal["blocker", "warning", "info"]
    message: str
    value: Any = None
    threshold: Any = None
    requires_human: bool = False      # a person should look at it before anything is traded, even in Auto mode

    @model_validator(mode="after")
    def _a_blocker_has_a_known_code(self):
        if self.severity == "blocker" and self.code not in BLOCKER_CODES[self.dimension]:
            raise ValueError(f"{self.code!r} is not a blocker code of {self.dimension}")
        return self


class Measure(_Model):
    """One piece of a dimension's score: `value` is 0 to 1 (what it earned of its weight), `available` False when its input could not be made (it then counts the unknown credit)."""
    name: str
    x: Any = None
    value: float
    weight: float
    available: bool = True
    note: str | None = None


class Dimension(_Model):
    weight: float
    score: float                      # 0 to 100
    points: float                     # score x weight / 100: what the scorecard prints ("accuracy 27/30")
    applicable: bool = True           # False: not part of the total (risk fit and value do not apply to a reduction)
    measures: list[Measure] = Field(default_factory=list)


class Recheck(_Model):
    """One value fetched again and compared with the one on the record."""
    name: str
    kind: str                         # bars | bars_changed | provider | provider_price_linked | filing | filing_changed | derived
    recorded: Any = None
    fetched: Any = None
    tolerance: float | None = None
    match: bool
    decisive: bool = False
    note: str | None = None


class SourceCheck(_Model):
    ref: str                          # [4], E15 or an event class
    chunk_id: str | None = None
    found: bool
    quote_in_text: bool
    same_text: bool | None = None
    tagged: bool | None = None        # the source read again lists this recommendation's symbol
    note: str | None = None


class CrossCheck(_Model):
    name: str                         # price | identity | high_52w
    a: Any = None
    b: Any = None
    tolerance: float | None = None
    passed: bool
    note: str | None = None


class Integrity(_Model):
    checked: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)


class Identity(_Model):
    listed: bool | None = None
    listed_company: str | None = None
    company: str | None = None


class AccuracyDetail(_Model):
    integrity: Integrity = Field(default_factory=Integrity)
    rechecked: list[Recheck] = Field(default_factory=list)
    sources: list[SourceCheck] = Field(default_factory=list)
    cross: list[CrossCheck] = Field(default_factory=list)
    identity: Identity = Field(default_factory=Identity)
    match_rate: float | None = None


class ClaimLabel(_Model):
    ref: str
    direction: str
    label: str
    reason: str | None = None         # the model's words, shortened


class Control(_Model):
    ref: str
    expected: str = "unrelated"
    label: str | None = None
    passed: bool


class FreshItem(_Model):
    id: str
    chunk_id: str | None = None
    headline: str | None = None
    published_utc: str | None = None
    corpus: str | None = None
    classes: list[str] = Field(default_factory=list)   # bearish event classes CODE matched in a sentence that names the company
    label: str = "none"                                # the model's: bearish (with a verified quote) or none
    quote: str | None = None


class Counterfactual(_Model):
    news: float | None = None
    filings: float | None = None
    blend: float | None = None                         # the Advisor's blend worked again with the review's labels applied
    blend_recorded: float | None = None                # the blend the Advisor recorded


class EvidenceDetail(_Model):
    reviewed: bool = False
    claims: list[ClaimLabel] = Field(default_factory=list)
    control: Control | None = None
    fresh: list[FreshItem] = Field(default_factory=list)
    counterfactual: Counterfactual | None = None


class TimingDetail(_Model):
    session: str
    trading_date: str | None = None
    session_reason: str = ""
    window_closes_utc: str | None = None
    next_open_utc: str | None = None
    record_date: str | None = None
    quote_price: float | None = None
    quote_as_of_utc: str | None = None
    deal_price: float | None = None
    price_basis: str | None = None                     # quote | last_close
    zone_position: float | None = None                 # 0 at the bottom of the entry zone, 1 at the top


class RiskDetail(_Model):
    tolerance_stated: str | None = None
    tolerance_effective: str | None = None
    allowed_level: str | None = None
    risk_score: float | None = None
    risk_level: str | None = None
    flags: list[str] = Field(default_factory=list)


class ValueDetail(_Model):
    deal: str | None = None                            # falling | unknown | overpriced | rich | great | good | fair
    avg_6m: float | None = None
    high_6m: float | None = None
    low_6m: float | None = None
    premium: float | None = None
    discount: float | None = None
    reward_to_risk: float | None = None
    earnings_yield: float | None = None
    bonus: float = 0.0


class PatternHit(_Model):
    name: str
    direction: str                                     # bullish | bearish
    points: float
    evidence: str                                      # the numbers that make it true, in words


class ChartDetail(_Model):
    patterns: list[PatternHit] = Field(default_factory=list)   # the patterns that hold
    unknown: list[str] = Field(default_factory=list)           # the patterns that could not be judged: a number they need could not be made
    points: float | None = None                                # the sum of the points of the patterns that hold
    facts: dict[str, Any] = Field(default_factory=dict)        # the numbers the Data Clerk gave, as the patterns read them


class Audit(_Model):
    critic_model: str | None = None
    advisor_model: str | None = None
    independent_model: bool | None = None              # the critic's model is from a different producer than the Advisor's; None when either is unknown
    config_version: str
    metric_order: str | None = None
    quote_order: str | None = None
    verify_order: str | None = None
    run: dict | None = None


class Review(_Model):
    schema_version: int = SCHEMA_VERSION
    review_id: str
    record_id: str
    request_id: str
    symbol: str
    action: str
    reviewed_utc: str
    valid_until_utc: str | None = None                 # approved reviews only: after it the approval is void (the price check is old)
    # the verdict
    approved: bool
    score: int
    score_before_blockers: int
    grade: str
    bonus: float = 0.0
    requires_human: bool = False
    requires_human_reasons: list[str] = Field(default_factory=list)
    denial_reasons: list[str] = Field(default_factory=list)
    summary: str
    blockers: list[Finding] = Field(default_factory=list)
    warnings: list[Finding] = Field(default_factory=list)
    notes: list[Finding] = Field(default_factory=list)
    dimensions: dict[str, Dimension]
    accuracy: AccuracyDetail
    evidence: EvidenceDetail
    timing: TimingDetail
    risk: RiskDetail
    value: ValueDetail
    chart: ChartDetail = Field(default_factory=ChartDetail)
    audit: Audit

    @property
    def findings(self) -> list[Finding]:
        return [*self.blockers, *self.warnings, *self.notes]
