"""The Account Manager's thresholds: `configs/account_manager_v1.json`, read and checked ONCE (design/sizing_in_account_manager_spec.md).

    config = load_config()                  # the file, validated: a missing key, an unknown key or a wrong type is an error NOW, not halfway through a purchase
    config.sizing.fractional_even_split     # true: an even split is in fractional shares (design choices entry 110); false: in whole shares
    config.clarify.max_questions            # 3: the Ask-mode questions about how much to buy (entry 85)
    config.direct.valid_minutes             # 30: how long the price of a trade the user named stays good (entry 87)
    config.ignore_critic.valid_minutes      # 30: the same for an Advisor recommendation the Critic did not review because the user ticked "Ignore Critic" (entry 148)
    config.missing.max_prompts              # 5: the most Ask-mode questions about stocks the Data Clerk did not find (entry 102)
    config.orders.watch_seconds             # 60: how often the web server's watcher checks the waiting limit orders while the market is open (entry 140)

Every threshold of the Account Manager's sizing is in that file and nowhere else. There is no size limit of the user's any more (design choices entry 91: the only check is that the funds cover the order). A model never
chooses a threshold and the code never writes one. `extra="forbid"` means a typo in the JSON fails when the file is loaded, instead of being ignored while the code uses a default (the same rule as the Critic's and the
Advisor's `config.py`).

`load_config()` keeps what it read: asking again for the same file returns the same object (the sections are frozen), until the file changes on disk.
"""
import functools
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from prometheus.rag.config import REPO_ROOT

CONFIG_PATH = REPO_ROOT / "configs" / "account_manager_v1.json"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Sizing(_Section):
    fractional_even_split: bool                      # design choices entry 110 (F9): True, a split evenly buys the same DOLLARS of each stock by buying a fraction of a share; False, the whole-share split of entry 108 (Part G). Round robin is whole shares either way


class Clarify(_Section):
    max_questions: int = Field(ge=1, le=5)           # the most questions the loop may put to the user (the spec: three)
    max_problems: int = Field(ge=0, le=5)            # the most replies of the model, or answers of the user, that code refuses (not JSON, an amount nobody typed, a number the account cannot pay) before the loop gives up and the default size is shown
    max_question_chars: int = Field(ge=40, le=1000)  # a question longer than this is cut (it is text from a model, shown to the user)
    max_answer_chars: int = Field(ge=20, le=1000)    # the user's answer is cut to this before the model reads it


class Missing(_Section):
    max_prompts: int = Field(ge=1, le=10)            # the most questions "OK to continue without it?" one order may need (one for each stock the Data Clerk did not find); more missing stocks than this is a request to clarify (entry 102)
    max_question_chars: int = Field(ge=80, le=1000)  # the model's wording of one such question may be this long at most: a longer one is refused and the code's own sentence is shown
    max_problems: int = Field(ge=0, le=3)            # the most times a wording that code refused is asked for again before the code's own sentence is shown
    max_seconds: int = Field(ge=1, le=300)           # a model that takes longer than this to word one question is given up on: the code's own sentence is shown, and the user is never kept waiting for it


class Direct(_Section):
    valid_minutes: int = Field(ge=1, le=240)         # a trade the user named has no review and so no `valid_until_utc` of the Critic's: the price the Account Manager asked of the Data Clerk is good for this long (entry 87)


class Neutral(_Section):
    valid_minutes: int = Field(ge=1, le=240)         # a neutral recommendation has no review and so no `valid_until_utc` of the Critic's: the price the Account Manager asked of the Data Clerk is good for this long (entry 91)


class IgnoreCritic(_Section):
    valid_minutes: int = Field(ge=1, le=240)         # an Advisor recommendation the user chose not to have the Critic review (entry 148, "Ignore Critic") has no review and so no `valid_until_utc` of the Critic's: the price the Account Manager asked of the Data Clerk is good for this long, as a neutral one's


class Orders(_Section):
    watch_seconds: int = Field(ge=5, le=3600)        # how often the watcher in the web server quotes the symbols of the waiting limit orders, DURING the regular session only (entry 140, W5): one Finnhub quote call for all of them, a convention of 60 seconds


class AccountManagerConfig(_Section):
    config_version: str = Field(min_length=1)
    sizing: Sizing
    clarify: Clarify
    missing: Missing
    direct: Direct
    neutral: Neutral
    ignore_critic: IgnoreCritic
    orders: Orders


@functools.cache
def _read(path: Path, stamp: int) -> AccountManagerConfig:
    """The file at `path`, read and checked; `stamp` (its modification time) is part of the cache key, so an edited file is read again."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"{path}: the Account Manager's settings file does not exist") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: not valid JSON ({error.msg})") from None
    try:
        return AccountManagerConfig.model_validate({k: v for k, v in raw.items() if not k.startswith("_")})
    except Exception as error:  # pydantic's ValidationError: its text names every bad key
        raise ValueError(f"{path}: {error}") from None


def load_config(path: str | Path | None = None) -> AccountManagerConfig:
    """Read and check the Account Manager's settings (the same object again while the file is unchanged). Raises ValueError, naming the file and the problem, when the file is missing, is not valid JSON or does not
    match (a missing, unknown or mistyped key)."""
    path = Path(path) if path else CONFIG_PATH
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        raise ValueError(f"{path}: the Account Manager's settings file does not exist") from None
    return _read(path, stamp)
