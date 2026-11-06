"""The quality multiplier: a company's weak numbers shrink what a strategy may expect from it (Advisor v1 spec section 9.3, design choices entry 51).

    quality = quality_multiplier(metrics, today, config.quality)
    quality.multiplier        # 1.0 times a factor for every check that FAILED, never below the floor (0.2)
    quality.haircuts          # [{check, value, factor}]  every factor applied, with the value that earned it
    quality.haircuts_skipped  # ["roe: Finnhub has no value ...", ...]  every check that could not run, and why
    quality.turned_off        # families a failing check switches off for this symbol (the earnings-quality checks turn off `reversion`)

The checks are a table in `configs/advisor_v1.json` (`quality.checks`): a metric, a comparison, a threshold, a factor. A MISSING input applies no factor and is listed in
`haircuts_skipped`: an unknown is not a defect, but it is visible, so the Critic can see how much of the multiplier was actually measured. The checks that read a figure of a
filing (`statement: true`) are skipped, with a note, when that filing is older than `statement_max_age_days` (120): a year-old balance sheet says little about today's accruals.
Checks with the same `group` apply their factor ONCE, however many of them fail (three signs of poor earnings quality are one finding).
"""
import operator
from datetime import date

from prometheus.agents.advisor.config import Quality as QualityConfig
from prometheus.agents.advisor.gates import WAIVED_NOTE, negative_equity, note, value
from prometheus.agents.advisor.record import Haircut, Quality

OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge, "==": operator.eq}


def quality_multiplier(metrics: dict | None, today: date, config: QualityConfig) -> Quality:
    """The multiplier for one symbol from its metrics (`data.metrics[symbol]`; None or empty: every check is skipped and the multiplier is 1.0)."""
    multiplier, haircuts, skipped, turned_off, groups_hit = 1.0, [], [], [], set()
    for check in config.checks:
        if check.metric == "debt_to_equity" and negative_equity(metrics):
            skipped.append(f"{check.name}: {WAIVED_NOTE}")
            continue
        got = value(metrics, check.metric)
        if got is None:
            skipped.append(f"{check.name}: {note(metrics, check.metric)}")
            continue
        as_of = (metrics.get(check.metric) or {}).get("as_of")
        if check.statement and as_of and (today - date.fromisoformat(as_of)).days > config.statement_max_age_days:
            skipped.append(f"{check.name}: the statement is from {as_of}, more than {config.statement_max_age_days} days old")
            continue
        if not OPS[check.op](got, check.value):
            continue
        for family in check.turns_off:
            if family not in turned_off:
                turned_off.append(family)
        if check.group and check.group in groups_hit:
            haircuts.append(Haircut(check=check.name, value=got, factor=1.0))  # another check of the group already took the factor: recorded, not applied twice
            continue
        if check.group:
            groups_hit.add(check.group)
        multiplier *= check.factor
        haircuts.append(Haircut(check=check.name, value=got, factor=check.factor))
    return Quality(multiplier=round(max(config.floor, multiplier), 6), haircuts=haircuts, haircuts_skipped=skipped, turned_off=turned_off)
