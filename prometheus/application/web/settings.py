"""The two checkboxes of the request card as the settings of ONE run (UI v1 spec 5.3).

    settings = run_settings(strict=False, neutral=True)
    with applied(settings):            # the environment variable is set for the length of the run and put back afterwards, even when the run raises
        ...
    settings.allow_neutral             # whether a neutral recommendation counts as approved in this run

CONFIRMED BY JACOB (2026-11-15; the spec's section 5.3): "The Strict Recommendations is only allow Approved recommendations (no neutral allowed)", and the neutral box is `ALLOW_NEUTRAL_RECOMMENDATIONS` for the run
("ALLOW_NEUTRAL_RECOMMENDATIONS=true in .env follows that matching checkbox": the box decides it; the value in `.env` is only what a run WITHOUT the page uses). THE WHOLE MAPPING IS `run_settings`, so changing it is a
few lines here and nowhere else.

    Strict off   the neutral box says whether a neutral recommendation counts as approved (`ALLOW_NEUTRAL_RECOMMENDATIONS`, which the Account Manager reads through `rag.config.setting`: the environment first).
    Strict on    only what the Critic approves trades: a neutral recommendation is NOT a pass whatever the other box says (the page shows that box unticked and disabled).

There is no other setting. Strict used to lower the automatic-trading mark too; the mark is gone (design choices entry 104: Auto mode trades every recommendation the Critic approves, and Strict does not change that).
Nothing is saved to the account. `.env` is never read or written by this module: the variable is set in this process's own environment and restored.
"""
import os
from contextlib import contextmanager
from dataclasses import dataclass

NEUTRAL_SETTING = "ALLOW_NEUTRAL_RECOMMENDATIONS"


@dataclass(frozen=True)
class RunSettings:
    allow_neutral: bool             # the value of ALLOW_NEUTRAL_RECOMMENDATIONS for the run: whether a neutral recommendation counts as approved


def run_settings(strict: bool, neutral: bool) -> RunSettings:
    """The settings of one run from the two checkboxes (see the module text): Strict means no neutral recommendation passes; otherwise the neutral box decides."""
    return RunSettings(allow_neutral=False if strict else bool(neutral))


@contextmanager
def applied(settings: RunSettings):
    """Set the run's environment variable and put it back as it was (set again, or removed) when the block ends, whatever happens inside it."""
    had = NEUTRAL_SETTING in os.environ
    before = os.environ.get(NEUTRAL_SETTING)
    os.environ[NEUTRAL_SETTING] = "true" if settings.allow_neutral else "false"
    try:
        yield settings
    finally:
        if had:
            os.environ[NEUTRAL_SETTING] = before
        else:
            os.environ.pop(NEUTRAL_SETTING, None)
