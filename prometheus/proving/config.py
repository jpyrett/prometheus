"""The Proving Ground's thresholds: `configs/proving_v1.json`, read and checked ONCE (design/proving_ground_spec.md section 5).

    config = load_config()                  # the file, validated: a missing key, an unknown key or a wrong type is an error NOW, not halfway through a run
    config.horizons                         # [5, 10, 20]      the sessions after the entry at which a recommendation's return is read
    config.benchmark                        # "SPY"
    config.min_sample                       # 20               under this many outcomes a rate or an average is shown with a small-sample mark and no colour of success or failure (P9)
    config.split_suspect_pct                # Decimal('0.40')  a close-to-open move past this is a suspected split (P10)
    config.bucket_of(73)                    # (70, 80)         the Advisor's score bucket, `[low, high)`: the last bucket's high is 101 so that a score of 100 is in it
    config.time_machine.cadences            # {"daily": 1, "twice_weekly": [0, 3], "weekly": [0], "every_two_weeks": 10}

Every threshold of the Proving Ground is in that file and nowhere else; the same rule as the Trader's, the Critic's and the Advisor's configs (`extra="forbid"`: a typo in the JSON fails when the file is loaded). A key that starts with
an underscore is a note and is dropped. A CADENCE is a whole number N (every N trading days) or a list of weekdays (0 is Monday, 4 is Friday: those days of every week); the engine of a later step reads them.
`split_suspect_pct` is a ratio that is compared with prices, so it is a `Decimal` of the digits the file writes it with (`money.Rate`, entry 112); `time_machine.cost_note_per_day_usd` is dollars (a `Decimal`, or null: no estimate is
shown) and is filled in from the measured cost of a real run, never guessed. A `_pct` is a fraction (0.40 is 40%), as everywhere in the desk.
"""
import json
from decimal import Decimal
from pathlib import Path

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator, model_validator

from prometheus.common.money import PositiveMoney, Rate
from prometheus.rag.config import REPO_ROOT
from prometheus.rag.loaders import check_symbol

CONFIG_PATH = REPO_ROOT / "configs" / "proving_v1.json"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")      # the reasoning efforts OpenRouter takes ("none" turns the reasoning off; a model honours only the ones it has: GLM 5.3 has low, high and max, Nemotron 3 Super only on and off)


class RoleSettings(_Section):
    """How ONE model is asked in the fast mode (design/proving_ground_spec.md P14): the same model, asked for less thinking and a shorter answer, and not waited for as long. `effort` None leaves the model's own (the live desk's)."""
    effort: Literal[EFFORTS] | None = None
    max_output_tokens: StrictInt | None = Field(default=None, gt=0)      # a cap on the reply (the reasoning counts in it): it must leave room for the JSON the agent checks
    timeout_seconds: float = Field(gt=0)                                 # the client's wait for one response
    max_retries: StrictInt = Field(ge=0)                                 # the client's own retries of one call (a 429, a 5xx, a timeout)


class ModeSettings(_Section):
    """What one mode (fast or faithful) asks of the models. NOTHING here is a rule of the desk: no score, gate, threshold, verdict, size or fill is read from this section."""
    provider_sort: Literal["throughput", "latency", "price"] | None = None      # OpenRouter's `provider.sort`: the SAME model served from the host that sorts first (null: OpenRouter's own routing, as live)
    parallel_records: StrictInt = Field(ge=1)                                   # the records of one review the Critic checks at once (the review that comes out is the sequential one)
    call_limit_seconds: float | None = Field(default=None, gt=0)                # the wall limit of one model call of the Advisor's calls (null: the live 300 seconds)
    run_seconds_note: float | None = Field(default=None, gt=0)                  # the MEASURED wall time of a cold run in this mode (the days are researched at once, so about the slowest day's); null shows no estimate
    cost_note_per_day_usd: PositiveMoney | None = None                          # the MEASURED cost of one decision day in this mode, to the cent (null: the section's own `cost_note_per_day_usd`)
    advisor: RoleSettings | None = None
    critic: RoleSettings | None = None
    data_clerk: RoleSettings | None = None


class TimeMachine(_Section):
    max_decision_days: StrictInt = Field(gt=0)           # the most decision days one Time Machine run may have
    cadences: dict[str, StrictInt | list[StrictInt]]     # name -> every N trading days, or the weekdays (0 Monday .. 4 Friday) a decision is made on
    default_cadence: str
    min_history_sessions: StrictInt = Field(gt=0)        # the stored bars must reach this many sessions before the start (the 252-session indicators)
    min_news_days: StrictInt = Field(gt=0)               # the stored news must reach this many days before the start
    follow_after_end: StrictBool                         # the walk goes on to the last stored bar after the last decision day, so the positions are followed
    use_trader_model: StrictBool                         # the Trader's model takes part in a run (false: code places the orders, free and repeatable)
    cost_note_per_day_usd: PositiveMoney | None          # the measured cost of one decision day; null shows no estimate
    parallel_days: StrictInt = Field(gt=0)               # the most decision days researched at once (a run researches all its days at once up to this)
    retry_429: StrictInt = Field(ge=0)                   # how many times a day whose research a provider's rate limit (429) or a bad model reply stopped is tried again (never more than 2 of them are tried: engine.research_all)
    pause_429_seconds: float = Field(ge=0)               # the first wait before such a retry; it grows with each
    news_coverage_min_sources: StrictInt = Field(ge=0)   # a decision day whose research found fewer news articles than this for its companies is "thin": the caveat news_coverage_thin
    spare_recommendations: StrictInt = Field(ge=0)       # phase 1 asks the Advisor for the number asked for PLUS this many a day (never more than the Advisor's own limit, `work_order.MAX_TOP_K`): the walk takes the first of them the Advisor's own cooldown does not catch, as a live Advisor, which ranks with its ledger, would have moved on to the next company (Critic v2 spec section 8)
    mode: Literal["fast", "faithful"]                    # the default speed of a run (a request may say its own): fast asks the models for less (RoleSettings), faithful asks exactly what the live desk asks
    fast: ModeSettings
    faithful: ModeSettings

    @field_validator("cadences")
    @classmethod
    def _cadences(cls, value: dict) -> dict:
        if not value:
            raise ValueError("time_machine.cadences needs at least one cadence")
        for name, spec in value.items():
            if isinstance(spec, list):
                if not spec or sorted(set(spec)) != sorted(spec) or any(not 0 <= day <= 4 for day in spec):
                    raise ValueError(f"cadence {name!r}: a list of weekdays has each of 0 (Monday) to 4 (Friday) once, not {spec!r}")
            elif spec < 1:
                raise ValueError(f"cadence {name!r}: a number of trading days is at least 1, not {spec!r}")
        return value

    @model_validator(mode="after")
    def _default_is_one_of_them(self):
        if self.default_cadence not in self.cadences:
            raise ValueError(f"time_machine.default_cadence {self.default_cadence!r} is not one of the cadences: {', '.join(self.cadences)}")
        return self

    @model_validator(mode="after")
    def _faithful_is_the_live_desk(self):
        faithful = self.faithful
        if faithful.provider_sort is not None or faithful.call_limit_seconds is not None or any(getattr(faithful, role) is not None for role in ("advisor", "critic", "data_clerk")):
            raise ValueError("time_machine.faithful asks the models exactly what the live desk asks: no provider_sort, no call_limit_seconds and no role settings (only parallel_records, which changes no answer)")
        return self


class Live(_Section):
    poll_seconds: StrictInt = Field(gt=0)
    only_in_session: StrictBool


class Store(_Section):
    root: str = Field(min_length=1)                      # relative to the repository (or absolute): where the results are written


class ProvingConfig(_Section):
    config_version: str = Field(min_length=1)
    horizons: list[StrictInt]                            # sessions after the entry, ascending
    benchmark: str
    min_sample: StrictInt = Field(gt=0)
    split_suspect_pct: Rate = Field(gt=0, lt=1)
    score_buckets: list[StrictInt]                       # the Advisor's score edges: [0, 40, 50 ...] makes [0, 40), [40, 50) ...
    time_machine: TimeMachine
    live: Live
    store: Store

    @field_validator("benchmark")
    @classmethod
    def _benchmark(cls, value: str) -> str:
        return check_symbol(value)

    @field_validator("horizons")
    @classmethod
    def _horizons(cls, value: list[int]) -> list[int]:
        if not value or any(h < 1 for h in value) or value != sorted(set(value)):
            raise ValueError(f"horizons are whole numbers of sessions of at least 1, ascending and each once, not {value!r}")
        return value

    @field_validator("score_buckets")
    @classmethod
    def _buckets(cls, value: list[int]) -> list[int]:
        if len(value) < 2 or value != sorted(set(value)) or value[0] != 0 or value[-1] < 101:
            raise ValueError(f"score_buckets are ascending edges from 0 to at least 101 (so that a score of 100 is in the last bucket), not {value!r}")
        return value

    @property
    def store_root(self) -> Path:
        """Where the results are written: `store.root`, made absolute from the repository."""
        root = Path(self.store.root)
        return root if root.is_absolute() else REPO_ROOT / root

    def buckets(self) -> list[tuple[int, int]]:
        """The Advisor's score buckets as (low, high) pairs, `[low, high)`."""
        return list(zip(self.score_buckets, self.score_buckets[1:]))

    def bucket_of(self, score) -> tuple[int, int] | None:
        """The (low, high) bucket an Advisor score falls in, or None for a score that is not a whole number inside them."""
        if isinstance(score, bool) or not isinstance(score, int):
            return None
        return next((b for b in self.buckets() if b[0] <= score < b[1]), None)


def load_config(path: str | Path | None = None) -> ProvingConfig:
    """Read and check the Proving Ground's settings. Raises ValueError, naming the file and the problem, when it is not valid JSON or does not match (a missing, unknown or mistyped key, a cadence or a bucket list that makes no sense)."""
    path = Path(path) if path else CONFIG_PATH
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"{path}: the Proving Ground's settings file does not exist") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: not valid JSON ({error.msg})") from None
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: the settings must be a JSON object")
    try:
        return ProvingConfig.model_validate({k: v for k, v in raw.items() if not k.startswith("_")})
    except Exception as error:  # pydantic's ValidationError: its text names every bad key
        raise ValueError(f"{path}: {error}") from None
