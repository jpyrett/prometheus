"""The Trader's thresholds: `configs/trader_v1.json`, read and checked ONCE (Trader v1 spec section 15).

    config = load_config()                  # the file, validated: a missing key, an unknown key or a wrong type is an error NOW, not halfway through a run
    config.orders.max_chase_pct             # 0.01
    paper_trading_enabled()                 # PAPER_TRADING from the environment, then .env: true, yes, on or 1 pass; anything else refuses every run

Every threshold of the Trader is in that file and nowhere else. The model never chooses one and the code never writes one. `extra="forbid"` means a typo in the JSON fails when the file is loaded, instead of
being ignored while the code uses a default (the same rule as the Critic's `config.py`). `mode` is `paper` or `backtest` and nothing else: live trading is refused here, in the loader, and again by the Trader.
`cash_policy` and `fill_at` each have one value in this version, and `commission_per_order` must be 0. The keys exist so that a later version can add a value; this version refuses any other, so that a file cannot ask for
behaviour the code does not have. (A commission is refused because `apply_fill` and the account book charge none: the broker would hold back cash for it that the account never pays, and the cash the Trader reports
would not be the cash the book holds. The broker and the simulator still carry the commission in their arithmetic, ready for the version that charges one; here it is always 0.)
"""
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator, model_validator

from prometheus.rag.config import REPO_ROOT, setting
from prometheus.rag.loaders import check_symbol

CONFIG_PATH = REPO_ROOT / "configs" / "trader_v1.json"
MODES = ("paper", "backtest")            # `live` is not a mode of this version
CASH_POLICIES = ("reduce",)              # a purchase the cash does not fully cover is cut to the shares it can pay for
FILL_AT = ("next_open",)                 # a backtest enters at the open of the first session after the decision
TRUE_WORDS = ("true", "yes", "on", "1")  # the values of PAPER_TRADING that let the Trader run


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Limits(_Section):
    max_trades: StrictInt = Field(gt=0)                 # the most tickets one run places (decisions that permit a trade, funding sales included)
    base_turns: StrictInt = Field(gt=0)                 # the model's turns when there is nothing to place; one more for each ticket still `new`
    max_turns: StrictInt = Field(gt=0)                  # and never more than this
    model_timeout_seconds: float = Field(gt=0)          # how long the client waits for one response of the Trader's model
    max_note_words: StrictInt = Field(gt=0)             # a trade's `note` (the model's words) is cut to this many words
    max_summary_words: StrictInt = Field(gt=0)          # the run's `summary` (the model's words) may be at most this long, or code writes its own

    @model_validator(mode="after")
    def _check(self):
        if self.max_turns < self.base_turns:
            raise ValueError("limits.max_turns must be at least limits.base_turns")
        return self


class Orders(_Section):
    max_chase_pct: float = Field(ge=0, lt=1)            # how far past the price the Critic judged an order may fill (a buy above it, a sale below it)
    limit_band_pct: float = Field(ge=0, lt=1)           # how far the other way a limit price may be set
    allow_market: StrictBool                            # false: only limit orders
    cash_policy: str

    @field_validator("cash_policy")
    @classmethod
    def _policy(cls, value: str) -> str:
        if value not in CASH_POLICIES:
            raise ValueError(f"orders.cash_policy must be {' or '.join(map(repr, CASH_POLICIES))} (this version has no other); got {value!r}")
        return value


class Costs(_Section):
    slippage_bps: float = Field(ge=0, le=1000)          # basis points: a buy fills this much above the price, a sale this much below
    commission_per_order: float = Field(ge=0)           # dollars per filled order: must be 0 in this version (see the validator)

    @field_validator("commission_per_order")
    @classmethod
    def _no_commission(cls, value: float) -> float:
        if value != 0:
            raise ValueError("costs.commission_per_order must be 0 in this version: the portfolio's apply_fill and the account book charge no commission, so any other value would count cash "
                             f"the account never pays and the Trader's cash would not match the book's; got {value!r}")
        return value


class Funding(_Section):
    require_user_approval: StrictBool                   # the sells that pay for a purchase need the user's yes (Ask and Auto alike)


class Backtest(_Section):
    fill_at: str
    max_follow_days: StrictInt = Field(gt=0, le=400)    # calendar days of bars after the decision that a trade is followed through (the bar request covers at most 400 days)
    benchmark: str                                      # the symbol a trade's return is compared with

    @field_validator("fill_at")
    @classmethod
    def _fill_at(cls, value: str) -> str:
        if value not in FILL_AT:
            raise ValueError(f"backtest.fill_at must be {' or '.join(map(repr, FILL_AT))} (this version has no other); got {value!r}")
        return value

    @field_validator("benchmark")
    @classmethod
    def _benchmark(cls, value: str) -> str:
        return check_symbol(value)


class Ledger(_Section):
    path: str = Field(min_length=1)                     # the trades ledger: one line per trade that became filled or rejected (paper mode only), relative to the repository


class TraderConfig(_Section):
    config_version: str = Field(min_length=1)
    mode: str                                           # the default mode of a run: paper or backtest
    limits: Limits
    orders: Orders
    costs: Costs
    funding: Funding
    backtest: Backtest
    ledger: Ledger

    @field_validator("mode")
    @classmethod
    def _mode(cls, value: str) -> str:
        if value not in MODES:
            raise ValueError(f"mode must be {' or '.join(MODES)} (live trading is refused in this version); got {value!r}")
        return value


def load_config(path: str | Path | None = None) -> TraderConfig:
    """Read and check the Trader's settings. Raises ValueError, naming the file and the problem, when it is not valid JSON or does not match (a missing, unknown or mistyped key, a mode or a policy this version refuses)."""
    path = Path(path) if path else CONFIG_PATH
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"{path}: the Trader's settings file does not exist") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: not valid JSON ({error.msg})") from None
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: the settings must be a JSON object")
    try:
        return TraderConfig.model_validate({k: v for k, v in raw.items() if not k.startswith("_")})
    except Exception as error:  # pydantic's ValidationError: its text names every bad key
        raise ValueError(f"{path}: {error}") from None


def paper_trading_enabled() -> bool:
    """Whether PAPER_TRADING lets the Trader run (rule T4): the environment variable, then `.env`, then "true". true, yes, on and 1 (any case) pass; anything else, a typo included, is a no."""
    return setting("PAPER_TRADING", "true").strip().lower() in TRUE_WORDS
