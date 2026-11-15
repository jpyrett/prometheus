"""The Critic's thresholds: `configs/critic_v1.json`, read and checked ONCE (Critic v1 spec Appendix A, design choices entry 59).

    config = load_config()                  # the file, validated: a missing key, an unknown key or a wrong type is an error NOW, not halfway through a review
    config.verdict.approve_min_score        # 70

Every threshold, weight and tolerance of the Critic is in that file and nowhere else. The model never chooses one and the code never writes one. `extra="forbid"` means a typo in the JSON fails
when the file is loaded, instead of being ignored while the code uses a default (the same rule as the Advisor's `config.py`).
"""
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from prometheus.common.user_profile import TOLERANCES
from prometheus.rag.config import REPO_ROOT

CONFIG_PATH = REPO_ROOT / "configs" / "critic_v1.json"
DIMENSIONS = ("accuracy", "evidence", "risk_fit", "timing", "value", "chart")
# The chart patterns the Critic knows and the side each is on (`chart.py` judges them; the config gives each its points: positive for a bullish one, negative for a bearish one)
PATTERNS = {"golden_cross": "bullish", "bull_alignment": "bullish", "uptrend_structure": "bullish", "breakout_on_volume": "bullish", "near_high": "bullish", "macd_positive": "bullish",
            "healthy_rsi": "bullish", "obv_rising": "bullish", "death_cross": "bearish", "bear_alignment": "bearish", "downtrend_structure": "bearish", "dead_cat_bounce": "bearish",
            "overbought": "bearish", "macd_negative": "bearish", "obv_falling": "bearish"}
LEVELS = ("low", "medium", "high", "very_high")           # from the least risk to the most: the order matters
LABELS = ("supports", "weak", "unrelated", "contradicts", "unreviewed")
KINDS = ("bars", "bars_changed", "provider", "provider_price_linked", "filing", "filing_changed", "derived")
SESSION_STATES = ("regular", "pre_market", "after_hours")


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _adds_up(name: str, values, total: float = 100.0) -> None:
    if abs(sum(values) - total) > 1e-9:
        raise ValueError(f"{name} must add up to {total:g}, not {sum(values):g}")


class Limits(_Section):
    max_records: int = Field(gt=0)
    max_metrics: int = Field(gt=0)
    max_fresh: int = Field(gt=0)
    max_model_calls: int = Field(ge=0)
    model_timeout_seconds: float = Field(gt=0)     # how long the client waits for one response of the critic's model (it reasons before it answers)
    max_source_chars: int = Field(ge=500)          # the most of one source's text the model is shown (the part around the quote)


class Verdict(_Section):
    approve_min_score: int = Field(ge=0, le=100)
    blocked_score_cap: int = Field(ge=0, le=100)   # a recommendation with a blocker never shows more than this
    auto_min_score: int = Field(ge=0, le=100)      # Auto mode: the lowest score that trades without asking the user (the profile may raise or lower it)
    grades: dict[str, int]
    unknown_credit: float = Field(ge=0, le=1)      # what a measure that could not be made counts as, 0 to 1

    @model_validator(mode="after")
    def _check(self):
        if set(self.grades) != {"A", "B", "C", "D"}:
            raise ValueError("verdict.grades must have exactly A, B, C and D (anything under D is F)")
        if not self.grades["A"] > self.grades["B"] > self.grades["C"] > self.grades["D"] >= 0:
            raise ValueError("verdict.grades must fall from A to D")
        if self.blocked_score_cap >= self.approve_min_score:
            raise ValueError("verdict.blocked_score_cap must be under approve_min_score, or a blocked recommendation could be approved by its score")
        return self


class Weighted(_Section):
    """Parts of a score, each a share of 1: metrics, sources and cross-checks (accuracy), support, clean and breadth (evidence)."""
    metrics: float | None = None
    sources: float | None = None
    cross: float | None = None
    support: float | None = None
    clean: float | None = None
    breadth: float | None = None


class Accuracy(_Section):
    min_match_rate: float = Field(gt=0, le=1)
    parts: Weighted
    tolerance: dict[str, float]
    abs_floor: dict[str, float]
    price_linked: list[str]
    cross_price_rel: float = Field(gt=0)
    cross_high_52w_rel: float = Field(gt=0)
    max_feature_age_days: int = Field(ge=0)

    @model_validator(mode="after")
    def _check(self):
        _adds_up("accuracy.parts", [self.parts.metrics or 0, self.parts.sources or 0, self.parts.cross or 0], 1.0)
        missing = [k for k in KINDS if k not in self.tolerance]
        if missing or not {"bars", "filing", "provider", "provider_price_linked", "derived"} <= set(self.abs_floor):
            raise ValueError(f"accuracy.tolerance needs {', '.join(KINDS)} and accuracy.abs_floor needs bars, filing, provider, provider_price_linked and derived")
        return self


class ControlText(_Section):
    headline: str = Field(min_length=1)
    text: str = Field(min_length=20)


class Evidence(_Section):
    parts: Weighted
    label_weight: dict[str, float]
    fresh_since_days: int = Field(ge=1, le=30)
    fresh_k: int = Field(ge=1, le=20)
    fresh_event_max_age_days: int = Field(ge=0)
    breadth_full_at: int = Field(gt=0)
    old_evidence_days: int = Field(ge=0)
    max_reason_words: int = Field(gt=0)
    control_texts: list[ControlText] = Field(min_length=1)
    blocking_classes: list[str]

    @model_validator(mode="after")
    def _check(self):
        _adds_up("evidence.parts", [self.parts.support or 0, self.parts.clean or 0, self.parts.breadth or 0], 1.0)
        if set(self.label_weight) != set(LABELS):
            raise ValueError(f"evidence.label_weight needs exactly {', '.join(LABELS)}")
        if any(not 0 <= w <= 1 for w in self.label_weight.values()):
            raise ValueError("evidence.label_weight values must be from 0 to 1")
        return self


class Timing(_Section):
    max_quote_age_minutes: float = Field(gt=0)
    review_ttl_minutes: float = Field(gt=0)
    session_factor: dict[str, float]
    top_of_zone_factor: float = Field(ge=0, le=1)
    below_zone_factor: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _check(self):
        if set(self.session_factor) != set(SESSION_STATES):
            raise ValueError(f"timing.session_factor needs exactly {', '.join(SESSION_STATES)} (a closed market is 0)")
        return self


class AgeCap(_Section):
    min_age: int = Field(ge=18)
    tolerance: str


class HorizonCap(_Section):
    under_years: float = Field(gt=0)
    tolerance: str


class RiskMeasure(_Section):
    name: str
    safe: float | None = None
    risky: float | None = None
    weight: float = Field(gt=0)
    abs: bool = False          # of the absolute value
    log: bool = False          # of log10 (the bounds as well)


class Risk(_Section):
    levels: dict[str, float]           # the upper bound (a risk score under it) of low, medium and high
    allowed: dict[str, str]            # the highest level each tolerance accepts
    age_caps: list[AgeCap]
    horizon_caps: list[HorizonCap]
    unknown_points: float = Field(ge=0, le=1)
    flag_step: float = Field(gt=0, le=1)
    weak_quality: float = Field(gt=0, le=1)
    measures: list[RiskMeasure]

    @model_validator(mode="after")
    def _check(self):
        if set(self.levels) != {"low", "medium", "high"} or not 0 < self.levels["low"] < self.levels["medium"] < self.levels["high"] <= 100:
            raise ValueError("risk.levels needs low < medium < high, each from 0 to 100")
        if set(self.allowed) != set(TOLERANCES) or any(level not in ("low", "medium", "high") for level in self.allowed.values()):
            raise ValueError("risk.allowed needs conservative, moderate and aggressive, each a level of low, medium or high (very_high is allowed to nobody)")
        for cap in [*self.age_caps, *self.horizon_caps]:
            if cap.tolerance not in TOLERANCES:
                raise ValueError(f"risk caps name a tolerance that does not exist: {cap.tolerance}")
        _adds_up("risk.measures weights", [m.weight for m in self.measures])
        for m in self.measures:
            if m.name != "flags" and (m.safe is None or m.risky is None or m.safe == m.risky):
                raise ValueError(f"risk measure {m.name} needs different safe and risky bounds")
            if m.log and not (m.safe and m.safe > 0 and m.risky and m.risky > 0):
                raise ValueError(f"risk measure {m.name} is logarithmic: its bounds must be positive")
        return self


class ValueMeasure(_Section):
    name: str
    zero_at: float
    full_at: float
    weight: float = Field(gt=0)


class Bonus(_Section):
    max_points: float = Field(ge=0)
    zero_at: float
    full_at: float
    min_quality: float = Field(ge=0, le=1)


class Value(_Section):
    rich_premium: float
    max_premium: float
    good_discount: float
    great_discount: float
    min_earnings_yield: float
    max_ps_when_unprofitable: float
    knife_ret_21d: float
    knife_drawdown_63d: float
    bonus: Bonus
    measures: list[ValueMeasure]
    last_recommendation_days: int = Field(gt=0)
    last_recommendation_rise: float
    add_above_cost: float

    @model_validator(mode="after")
    def _check(self):
        _adds_up("value.measures weights", [m.weight for m in self.measures])
        if not self.rich_premium < self.max_premium:
            raise ValueError("value.rich_premium must be under value.max_premium")
        if not self.good_discount < self.great_discount:
            raise ValueError("value.good_discount must be under value.great_discount")
        if self.bonus.zero_at >= self.bonus.full_at:
            raise ValueError("value.bonus.zero_at must be under value.bonus.full_at")
        return self


class PatternPoints(_Section):
    name: str
    points: float


class DeadCat(_Section):
    min_decline: float = Field(lt=0)       # the fall from the peak close to the lowest close after it: at least this bad (a negative number)
    min_rebound: float = Field(gt=0)       # the bounce off that lowest close: at least this much
    max_trough_age: int = Field(gt=0)      # the lowest close was no more than this many sessions ago
    max_vs_peak: float = Field(lt=0)       # and the price is still at least this far under the peak (a negative number)


class Chart(_Section):
    neutral: float = Field(ge=0, le=100)   # the score of a chart with no pattern: the points of the patterns that hold move it up or down from here
    weak_below: float = Field(ge=0, le=100)
    volume_ratio_min: float = Field(gt=0)
    near_high: float = Field(gt=0, le=1)
    rsi_healthy_low: float
    rsi_healthy_high: float
    rsi_overbought: float
    dead_cat: DeadCat
    patterns: list[PatternPoints]

    @model_validator(mode="after")
    def _check(self):
        names = [p.name for p in self.patterns]
        if sorted(names) != sorted(PATTERNS):
            raise ValueError(f"chart.patterns needs exactly one entry for each of {', '.join(PATTERNS)}")
        for p in self.patterns:
            if (p.points > 0) != (PATTERNS[p.name] == "bullish") or p.points == 0:
                raise ValueError(f"chart pattern {p.name} is {PATTERNS[p.name]}: its points must be {'positive' if PATTERNS[p.name] == 'bullish' else 'negative'}")
        if not self.rsi_healthy_low < self.rsi_healthy_high < self.rsi_overbought:
            raise ValueError("chart needs rsi_healthy_low < rsi_healthy_high < rsi_overbought")
        return self


class Ledger(_Section):
    path: str


class CriticConfig(_Section):
    model_config = ConfigDict(extra="forbid", frozen=True)

    config_version: str
    limits: Limits
    weights: dict[str, float]
    verdict: Verdict
    accuracy: Accuracy
    evidence: Evidence
    timing: Timing
    risk: Risk
    value: Value
    chart: Chart
    ledger: Ledger

    @model_validator(mode="after")
    def _check(self):
        if set(self.weights) != set(DIMENSIONS):
            raise ValueError(f"weights needs exactly {', '.join(DIMENSIONS)}")
        _adds_up("weights", self.weights.values())
        return self


def load_config(path: str | Path | None = None) -> CriticConfig:
    """Read and check the Critic's settings. Raises ValueError, naming the file and the problem, when it is not valid JSON or does not match (a missing, unknown or mistyped key, weights that do
    not add up, a threshold that contradicts another)."""
    path = Path(path) if path else CONFIG_PATH
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"{path}: the Critic's settings file does not exist") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: not valid JSON ({error.msg})") from None
    try:
        return CriticConfig.model_validate({k: v for k, v in raw.items() if not k.startswith("_")})
    except Exception as error:  # pydantic's ValidationError: its text names every bad key
        raise ValueError(f"{path}: {error}") from None
