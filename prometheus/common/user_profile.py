"""The user's profile: how much risk they accept, and whether the desk trades on its own (Critic v1 spec section 4.1, design choices entry 56).

    profile = load_profile("configs/profiles/jacob_pyrett.json")
    profile.risk_tolerance            # "moderate"
    profile.trade_mode                # "ask": the user approves each trade; "auto": the desk trades what the Critic approves
    view = profile.risk_view()        # what the Critic may see: the tolerance, age, horizon, exclusions and minimum price, and no size limit and no money
    limits = profile.sizing_limits()  # the four size limits the Account Manager sizes a purchase by (agents/account_manager/sizing.py)

The JSON files are still what the desk loads by default. The account book (design choices entry 72) loads this same class from Postgres: the risk levels it can set are `TOLERANCES` below. Nothing here
calls an API or a model.

THE CRITIC SEES A `RiskView`, NOT THE PROFILE. The view holds what the Critic needs to judge a COMPANY for this user (tolerance, age, horizon, the exclusions and the minimum price) and leaves out the name, the mode, the
automatic-trading mark and every SIZE limit: whether the desk asks or trades on its own is the Account Manager's business, and a Critic's verdict must not depend on it (a recommendation is as good or as bad in Ask mode
as in Auto). How much to buy is the Account Manager's too (`SizingLimits`): `RiskView` forbids extra fields, so a size limit cannot even be put into it, and the Critic cannot know the user's money.
"""
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from prometheus.rag.loaders import check_symbol

TOLERANCES = ("conservative", "moderate", "aggressive")   # from the least risk to the most: the order matters (the Critic takes the lowest of several caps)
Tolerance = Literal["conservative", "moderate", "aggressive"]


class RiskView(BaseModel):
    """What the Critic is allowed to know about the user: what it needs to judge a company's risk, never the name, the trading mode or a size limit (how much to buy is the Account Manager's: `SizingLimits`)."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    risk_tolerance: Tolerance
    age: int | None = None
    investment_horizon_years: float | None = None
    min_price: float
    excluded_symbols: tuple[str, ...] = ()
    excluded_sectors: tuple[str, ...] = ()


class SizingLimits(BaseModel):
    """The user's size limits, which the Account Manager reads to work out how many shares to buy (`agents/account_manager/sizing.py`): the four numbers the profile holds. The Critic's `RiskView` has none of them."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_position_pct: float = Field(gt=0, le=1)                      # the most one stock may be of equity, after the purchase
    max_loss_per_trade_pct: float = Field(gt=0, le=1)                # the most a stop-out may lose, of equity
    max_new_exposure_pct: float = Field(gt=0, le=1)                  # the most all the purchases of ONE run may add, of equity
    max_order_dollars: float | None = Field(default=None, gt=0)      # a cap on one order; None: no cap

    def changed(self, changes: dict) -> "SizingLimits":
        """These limits with `changes` ({limit name: new value}) applied, checked the same way: for a limit the user raised or lowered for ONE purchase. A name that is not one of the four, or a value outside its range, is a ValueError."""
        return type(self).model_validate({**self.model_dump(), **changes})


class UserProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    user: str = Field(min_length=1)                         # must equal the portfolio's user: the Account Manager refuses a pair that does not match
    risk_tolerance: Tolerance
    age: int | None = Field(default=None, ge=18, le=120)    # lowers the tolerance (Critic spec 8.4); None: no cap
    investment_horizon_years: float | None = Field(default=None, gt=0)
    trade_mode: Literal["ask", "auto"] = "ask"              # ask: the user approves each trade. auto: the desk trades what the Critic approves
    auto_min_score: int | None = Field(default=None, ge=0, le=100)   # auto mode only: the lowest Critic score that trades without asking; None: the Critic's own `verdict.auto_min_score`
    max_position_pct: float = Field(default=0.10, gt=0, le=1)        # the most one stock may be of equity, after the trade
    max_loss_per_trade_pct: float = Field(default=0.01, gt=0, le=1)  # the most a stop-out may lose, of equity
    max_new_exposure_pct: float = Field(default=0.30, gt=0, le=1)    # the most all the trades approved in ONE review may add, of equity
    max_order_dollars: float | None = Field(default=None, gt=0)      # a cap on one order
    min_price: float = Field(default=5.0, ge=0)                      # no stock under this price
    excluded_symbols: list[str] = Field(default_factory=list)
    excluded_sectors: list[str] = Field(default_factory=list)        # sector ETF tickers (the keys of configs/sector_etfs.json)
    note: str | None = None

    @field_validator("excluded_symbols")
    @classmethod
    def _symbols(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(check_symbol(v.strip().lstrip("$")) for v in values))

    @field_validator("excluded_sectors")
    @classmethod
    def _sectors(cls, values: list[str]) -> list[str]:
        from prometheus.agents.data_clerk.metrics_catalog import sector_etfs  # imported here: the catalogue needs pyarrow and httpx, and a profile is plain data
        clean = list(dict.fromkeys(v.strip().upper() for v in values))
        unknown = [v for v in clean if v not in sector_etfs()]
        if unknown:
            raise ValueError(f"unknown sector ETF {', '.join(unknown)}; known sectors: {', '.join(sector_etfs())}")
        return clean

    def risk_view(self) -> RiskView:
        """What the Critic may see (see the module text): no size limit."""
        return RiskView(risk_tolerance=self.risk_tolerance, age=self.age, investment_horizon_years=self.investment_horizon_years, min_price=self.min_price,
                        excluded_symbols=tuple(self.excluded_symbols), excluded_sectors=tuple(self.excluded_sectors))

    def sizing_limits(self) -> SizingLimits:
        """The four size limits, for the Account Manager's sizing (see `SizingLimits`)."""
        return SizingLimits(max_position_pct=self.max_position_pct, max_loss_per_trade_pct=self.max_loss_per_trade_pct, max_new_exposure_pct=self.max_new_exposure_pct,
                            max_order_dollars=self.max_order_dollars)


def load_profile(path: str | Path) -> UserProfile:
    """Read a profile JSON file. Raises ValueError, with the reason, if the file is missing, not valid JSON or not a valid profile."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"{path}: the profile file does not exist") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: not valid JSON ({error.msg})") from None
    try:
        return UserProfile.model_validate(data)
    except ValueError as error:  # pydantic's ValidationError is a ValueError: name the file as well
        raise ValueError(f"{path}: {error}") from None
