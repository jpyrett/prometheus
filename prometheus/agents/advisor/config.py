"""The Advisor's thresholds: `configs/advisor_v1.json`, read and checked ONCE (Advisor v1 spec section 6, design choices entry 51).

    config = load_config()                      # the file, validated: a missing key, an unknown key or a wrong type is an error NOW, not halfway through a run
    config.gates.max_bar_age_sessions           # 1

Every number the strategies use is in that file and nowhere else: the model never chooses a threshold, a weight, a price, a size or a rank, and neither does the ledger (it may change
the `basis` of a family's prior to "measured", never a threshold). `extra="forbid"` means a typo in the JSON ("max_bar_age_sesions") fails when the file is loaded, instead of being
silently ignored while the code uses a default.
"""
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from prometheus.rag.config import REPO_ROOT

CONFIG_PATH = REPO_ROOT / "configs" / "advisor_v1.json"
FAMILIES = ("event", "trend", "reversion", "capital_return", "theme", "conviction")
REGIMES = ("trend", "narrow", "stress", "recovering", "unknown")


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Limits(_Section):
    max_candidates: int = Field(gt=0)
    max_companies: int = Field(gt=0)
    max_signals_per_kind: int = Field(gt=0)
    max_events: int = Field(gt=0)
    discover_min_articles: int = Field(gt=0)
    max_candidates_on_order: int = Field(gt=0)
    recent_bars: int = Field(ge=12)


class Gates(_Section):
    max_bar_age_sessions: int
    max_fundamentals_age_days: int
    red_flag_event_bars: int
    red_flag_debt_to_equity: float
    red_flag_current_ratio: float
    earnings_days: int
    earnings_estimate_low: int
    earnings_estimate_high: int


class Check(_Section):
    name: str
    metric: str
    op: str = Field(pattern=r"^(<|<=|>|>=|==)$")
    value: float | bool
    factor: float = Field(gt=0, le=1)
    group: str | None = None          # checks of one group apply their factor once, when any of them fails
    statement: bool = False           # reads a figure of a filing: skipped with a note when the filing is older than `statement_max_age_days`
    turns_off: list[str] = Field(default_factory=list)  # families this failing check switches off for the symbol


class Quality(_Section):
    floor: float = Field(gt=0, le=1)
    statement_max_age_days: int
    checks: list[Check]


class Trend(_Section):
    horizon: int
    prior_20d: float
    basis_ref: str
    min_mom_12_1: float
    min_pct_of_52w_high: float
    max_ret_21d: float
    max_rsi: float
    max_pct_b: float
    breakout_min_mom_12_1: float
    breakout_min_pct_of_52w_high: float
    max_ret_21d_uptrend: float
    base: float
    w_mom: float
    mom_scale: float
    w_macd: float
    w_adx: float
    adx_min: float
    w_obv: float
    w_lowvol: float
    lowvol: float


class Event(_Section):
    horizon: int
    prior_surprise: float
    prior_event: float
    basis_ref: str
    max_age_bars: int
    max_reaction_atr: float
    base: float
    w_unpriced: float
    w_volume: float
    volume_ratio_min: float
    sue_min: float


class Reversion(_Section):
    horizon: int
    prior_20d: float
    basis_ref: str
    ret_5d: float
    zscore: float
    rsi: float
    residual_5d: float
    sector_ret_5d_min: float
    spy_ret_5d_min: float
    roe: float
    debt_to_equity: float
    current_ratio: float
    revenue_growth: float
    base: float
    w_zscore: float
    zscore_floor: float
    zscore_span: float
    w_rsi: float
    w_pct_b: float
    w_volume: float
    volume_ratio: float
    w_candle: float
    cause_lookback_bars: int
    fundamental_causes: list[str]


class CapitalReturn(_Section):
    horizon: int
    prior_20d: float
    basis_ref: str
    min_share_count_change: float
    max_debt_to_equity: float
    max_payout_coverage: float
    min_revenue_growth: float
    base: float
    w_buyback: float
    w_shares: float
    shares_bonus: float
    w_event: float
    event_bars: int
    w_growth: float


class Theme(_Section):
    horizon: int
    prior_20d: float
    basis_ref: str
    min_articles: int
    min_positive_share: float
    min_quality: float
    base: float
    w_article: float
    article_scale: float
    w_laggard: float
    w_positive: float
    positive_high: float
    w_volume: float


class Conviction(_Section):
    """The blend of news, performance, historical metrics and filings as a strategy of its own (Amendment 5)."""
    horizon: int
    prior_20d: float
    basis_ref: str
    min_blend: float          # the blend (-1 to +1) must be at least this
    min_parts: int            # at least this many of the four blend parts must have a value
    min_news: float           # the news part, when it has a value, must be at least this (0: not negative)
    max_rsi: float            # not bought when RSI is this or more (a blow-off)
    score: float              # the family's score, fixed: the blend already scales the edge, so it is not counted twice


class Families(_Section):
    trend: Trend
    event: Event
    reversion: Reversion
    capital_return: CapitalReturn
    theme: Theme
    conviction: Conviction


class Regime(_Section):
    stress_vol: float
    narrow_breadth: float
    trend_breadth: float
    recovering_drawdown: float
    recovering_ret_21d: float
    min_spy_bars: int
    multipliers: dict[str, dict[str, float]]


class Ranking(_Section):
    diversity_within: float


class BlendWeights(_Section):
    news: float = Field(ge=0)          # news AND press releases, the newer the heavier
    performance: float = Field(ge=0)   # the stock's own price behaviour
    historical: float = Field(ge=0)    # the Finnhub history of the company: profitability, growth, valuation, balance sheet
    filings: float = Field(ge=0)       # what the SEC filings show: payouts, share count, cash flow, their evidence items

    def model_post_init(self, _):
        if abs(self.news + self.performance + self.historical + self.filings - 1.0) > 1e-9:
            raise ValueError("blend.weights must add up to 1")


class Measure(_Section):
    """One number that speaks for a group: `value = clip((x - neutral) / span, -1, 1)` (a negative span: lower is better), or for a yes/no metric +1 when it equals `good` and -1 when it does not
    (`only_bad`: only the bad state speaks, the good one is silence)."""
    group: str
    metric: str
    neutral: float | None = None
    span: float | None = None
    good: bool | None = None
    only_bad: bool = False
    waived_by_negative_equity: bool = False   # debt_to_equity: entry 54


class Blend(_Section):
    weights: BlendWeights
    half_life_days: float = Field(gt=0)       # a piece of news loses half its weight every this many days: weight = 0.5 ** (age_days / half_life_days)
    news_prior_n: float = Field(ge=0)         # shrinkage: the news value is sum(sign x weight) / (sum(weight) + prior), so one old quote is not certainty
    press_release_weight: float = Field(gt=0)  # a news article weighs 1.0; a company's own press release this much (its own words are not an independent voice)
    measures: list[Measure]


class Sleeve(_Section):
    sleeve: str
    top_k: int
    eligible: list[str]
    veto_only: list[str]


class Trade(_Section):
    """The trade plan's multipliers. `min_reward_to_risk` and `trailing_families` are CONVENTIONS (design/advisor_exit_plan_spec.md, rules E1 and E2): a target is at least `min_reward_to_risk` times the stop distance above the entry, and a family
    in `trailing_families` has no target and a stop that rises with the highest close. Neither is tuned on the outcomes (design choices entry 133)."""
    entry_atr: float
    stop_atr: dict[str, float]
    target_edge_multiple: float
    min_reward_to_risk: float = Field(ge=0)      # 0 is no floor: the target is the family's edge alone, as before Advisor config version 3
    trailing_families: list[str]
    trend_limit_atr: float
    event_limit_reaction_atr: float


class Neutral(_Section):
    """The EXIT PLAN of a neutral company (Advisor v2, `design/critic_v2_evidence_spec.md` section 7, `trade_plan.neutral_exit_plan`): two CONVENTIONS, not findings (nothing in the record says where a neutral company should go): the stop is
    `stop_atr_multiple` times the 14-day ATR under the last close, the time stop `horizon_bars` trading days after the decision day. They are Jacob's to change."""
    stop_atr_multiple: float = Field(gt=0)
    horizon_bars: int = Field(gt=0)


class Ledger(_Section):
    path: str


class AdvisorConfig(_Section):
    config_version: str
    note: str | None = Field(default=None, alias="_note")
    limits: Limits
    gates: Gates
    quality: Quality
    families: Families
    regime: Regime
    ranking: Ranking
    blend: Blend
    sleeves: dict[str, Sleeve]
    trade: Trade
    neutral: Neutral
    ledger: Ledger

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


def load_config(path: str | Path | None = None) -> AdvisorConfig:
    """Read and check the Advisor's settings. Raises ValueError, naming the file and the problem, when it is not valid JSON or does not match (a missing, unknown or mistyped key,
    a regime or family that has no multiplier)."""
    path = Path(path) if path else CONFIG_PATH
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"{path}: the Advisor's settings file does not exist") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: not valid JSON ({error.msg})") from None
    try:
        config = AdvisorConfig.model_validate(raw)
    except Exception as error:  # pydantic's ValidationError: its text names every bad key
        raise ValueError(f"{path}: {error}") from None
    for family in FAMILIES:
        missing = [r for r in REGIMES if r not in config.regime.multipliers.get(family, {})]
        if missing:
            raise ValueError(f"{path}: regime.multipliers.{family} has no value for {', '.join(missing)}")
        if family not in config.trade.stop_atr:
            raise ValueError(f"{path}: trade.stop_atr has no value for {family}")
    unknown_trailing = [f for f in config.trade.trailing_families if f not in FAMILIES]
    if unknown_trailing or len(set(config.trade.trailing_families)) != len(config.trade.trailing_families):
        raise ValueError(f"{path}: trade.trailing_families must name each strategy family once, not {config.trade.trailing_families}")
    for case, sleeve in config.sleeves.items():
        unknown = [f for f in [*sleeve.eligible, *sleeve.veto_only] if f not in FAMILIES]
        if case not in ("funds", "theme", "industry", "market") or unknown:
            raise ValueError(f"{path}: sleeves.{case} is not a case (funds, theme, industry, market) or names an unknown family {unknown}")
    return config
