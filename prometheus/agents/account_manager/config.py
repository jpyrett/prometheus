"""The Account Manager's thresholds: `configs/account_manager_v1.json`, read and checked ONCE (design/sizing_in_account_manager_spec.md).

    config = load_config()                  # the file, validated: a missing key, an unknown key or a wrong type is an error NOW, not halfway through a purchase
    config.sizing.max_adv_pct               # 0.05
    config.clarify.max_questions            # 3: the Ask-mode questions about how much to buy (entry 85)

Every threshold of the Account Manager's sizing is in that file and nowhere else. The user's OWN limits (how much of the account one stock may be, how much one stop-out may lose, how much one review may add,
the most for one order) are not here: they belong to the user and are in the profile (`UserProfile.sizing_limits()`). A model never chooses a threshold and the code never writes one. `extra="forbid"` means a typo in
the JSON fails when the file is loaded, instead of being ignored while the code uses a default (the same rule as the Critic's and the Advisor's `config.py`).

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
    max_adv_pct: float = Field(gt=0, le=1)          # the most of a stock's average daily dollar volume (20 days) one purchase may be: the liquidity cap
    float_guard: float = Field(gt=0, lt=1e-6)       # added before a share count is rounded down, so that 2.9999999999999996 shares of arithmetic is the 3 shares it is


class Clarify(_Section):
    max_questions: int = Field(ge=1, le=5)           # the most questions the loop may put to the user (the spec: three)
    max_problems: int = Field(ge=0, le=5)            # the most replies of the model, or answers of the user, that code refuses (not JSON, an amount nobody typed, a number the account cannot pay) before the loop gives up and the default size is shown
    max_question_chars: int = Field(ge=40, le=1000)  # a question longer than this is cut (it is text from a model, shown to the user)
    max_answer_chars: int = Field(ge=20, le=1000)    # the user's answer is cut to this before the model reads it


class AccountManagerConfig(_Section):
    config_version: str = Field(min_length=1)
    sizing: Sizing
    clarify: Clarify


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
