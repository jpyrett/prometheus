"""The checkboxes of the request card as the settings of ONE run (UI v1 spec 5.3; entry 142 added "Always raise funds").

    settings = run_settings(strict=False, neutral=True, raise_funds=False)
    with applied(settings):            # the environment variable is set for the length of the run and put back afterwards, even when the run raises
        ...
    settings.allow_neutral             # whether a neutral recommendation counts as approved in this run
    settings.always_raise_funds        # whether holdings may be sold to pay for a purchase the cash does not cover, as if "raise funds if needed" were in the request (ALWAYS_RAISE_FUNDS: entry 142, R12)

CONFIRMED BY JACOB (2026-11-15; the spec's section 5.3): "The Strict Recommendations is only allow Approved recommendations (no neutral allowed)", and the neutral box is `ALLOW_NEUTRAL_RECOMMENDATIONS` for the run
("ALLOW_NEUTRAL_RECOMMENDATIONS=true in .env follows that matching checkbox": the box decides it; the value in `.env` is only what a run WITHOUT the page uses). THE WHOLE MAPPING IS `run_settings`, so changing it is a
few lines here and nowhere else.

    Strict off   the neutral box says whether a neutral recommendation counts as approved (`ALLOW_NEUTRAL_RECOMMENDATIONS`, which the Account Manager reads through `rag.config.setting`: the environment first).
    Strict on    only what the Critic approves trades: a neutral recommendation is NOT a pass whatever the other box says (the page shows that box unticked and disabled).

    Always raise funds   the box is `ALWAYS_RAISE_FUNDS` for the run (the Account Manager reads it through `rag.config.setting`, like the neutral setting): off unless ticked, and what the server's `.env` says is only what a run WITHOUT the page uses.
                         It is the same as the words "raise funds if needed" being in the request; "don't sell anything" in the words of a request still wins over it (`funding.permission`). It does not change what Strict means.

There is no other setting. Strict used to lower the automatic-trading mark too; the mark is gone (design choices entry 104: Auto mode trades every recommendation the Critic approves, and Strict does not change that).
Nothing is saved to the account. `.env` is never read or written by this module: the variable is set in this process's own environment and restored.
"""
import os
from contextlib import contextmanager
from dataclasses import dataclass

NEUTRAL_SETTING = "ALLOW_NEUTRAL_RECOMMENDATIONS"
RAISE_FUNDS_SETTING = "ALWAYS_RAISE_FUNDS"


@dataclass(frozen=True)
class RunSettings:
    allow_neutral: bool             # the value of ALLOW_NEUTRAL_RECOMMENDATIONS for the run: whether a neutral recommendation counts as approved
    always_raise_funds: bool = False      # the value of ALWAYS_RAISE_FUNDS for the run (entry 142, R12): the same as "raise funds if needed" being in the request


def run_settings(strict: bool, neutral: bool, raise_funds: bool = False) -> RunSettings:
    """The settings of one run from the checkboxes (see the module text): Strict means no neutral recommendation passes; otherwise the neutral box decides; the raise-funds box is its own and off unless ticked."""
    return RunSettings(allow_neutral=False if strict else bool(neutral), always_raise_funds=bool(raise_funds))


@contextmanager
def applied(settings: RunSettings):
    """Set the run's environment variables and put them back as they were (set again, or removed) when the block ends, whatever happens inside it."""
    values = {NEUTRAL_SETTING: "true" if settings.allow_neutral else "false", RAISE_FUNDS_SETTING: "true" if settings.always_raise_funds else "false"}
    before = {name: os.environ.get(name) for name in values if name in os.environ}
    os.environ.update(values)
    try:
        yield settings
    finally:
        for name in values:
            if name in before:
                os.environ[name] = before[name]
            else:
                os.environ.pop(name, None)
