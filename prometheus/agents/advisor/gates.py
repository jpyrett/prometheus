"""The gates: a symbol that fails one is not recommended, and the failure is RECORDED, never silent (Advisor v1 spec section 9.1, design choices entry 51).

    result = run_gates(metrics, events, today=today, config=config.gates, last_earnings_release="2026-07-29")
    result.failed            # ["freshness"]
    result.blocks("trend")   # True

A gate is arithmetic over the Data Clerk's metrics and the checked events: plain Python, no model. A gate whose INPUT is missing FAILS, with the missing input named (`not computable: ...`):
a stock must not be recommended because its bars could not be read, and the fix is more data, not a guess.

THERE IS NO LIQUIDITY GATE (design choices entries 93 to 95; Jacob, 2026-11-15: "There are no liquidity gates at all for the Advisor"): nothing about how much a stock trades decides anything here, and no such number is asked for.
The user's own minimum price is the Critic's (`RiskView.min_price`), not a gate.

    freshness   all         the newest stored bar is within `max_bar_age_sessions` weekdays of today; the Finnhub figures are not older than `max_fundamentals_age_days`
    red_flag    all         a going-concern, material-weakness or restatement event in the last 60 bars; debt over 4x equity with a current ratio under 1;
                            cash from operations negative while profit is positive. A HELD symbol that fails it is still reviewed (the holdings pass may say `reduce`)
    event_risk  all         the next earnings report is 3 days away or less; without the calendar, the last results press release is 85 to 95 days old (estimated, and the note says so).
                            It blocks every family EXCEPT `event` (an earnings event is what that family trades)
    no_metrics  all         the Data Clerk returned nothing for the symbol
"""
from dataclasses import dataclass, field
from datetime import date, timedelta

from prometheus.agents.advisor.config import Gates as GateConfig
from prometheus.agents.advisor.record import Gate

RED_FLAG_EVENTS = ("going_concern", "material_weakness", "restatement")  # "delisting_notice" is parked with the other classes the stored text has no example of


WAIVED_NOTE = "debt_to_equity rule waived: negative stockholders' equity (the ratio has no meaning)"


def negative_equity(metrics: dict | None) -> bool:
    """True when the stored balance sheet shows a stockholders' deficit. Then every debt-to-equity rule is waived, and the result says so (design choices entry 54). An unknown (no stored balance sheet) is not True: the rules apply."""
    return value(metrics, "negative_equity") is True


def value(metrics: dict | None, name: str):
    """The value of one metric result, or None."""
    return ((metrics or {}).get(name) or {}).get("value")


def note(metrics: dict | None, name: str) -> str:
    """Why a metric is missing, in the Data Clerk's words."""
    return ((metrics or {}).get(name) or {}).get("note") or "missing"


def sessions_between(last: date, today: date) -> int:
    """The weekdays after `last` up to and including `today`: how many sessions the newest bar is behind (holidays are not known, so this can overstate by one a few days a year)."""
    day, count = last, 0
    while day < today:
        day += timedelta(days=1)
        count += day.weekday() < 5
    return count


@dataclass
class GateResult:
    gates: list[Gate] = field(default_factory=list)

    @property
    def failed(self) -> list[str]:
        return [g.gate for g in self.gates if not g.passed]

    def first_failed(self) -> Gate | None:
        return next((g for g in self.gates if not g.passed), None)

    def blocks(self, family: str) -> bool:
        """Whether a failed gate stops this family: any failed gate does, except that `event_risk` alone leaves the `event` family alone."""
        failed = self.failed
        if not failed:
            return False
        return not (family == "event" and set(failed) == {"event_risk"})


def run_gates(metrics: dict | None, events: list[dict], today: date, config: GateConfig, last_earnings_release: str | None = None) -> GateResult:
    """Every gate for one symbol. `metrics` is `data.metrics[symbol]` (None or empty: the `no_metrics` gate fails and the rest are not run), `events` the checked events with their
    `age_bars`."""
    result = GateResult()
    if not metrics:
        result.gates.append(Gate(gate="no_metrics", passed=False, value=None, threshold="a result for the symbol", note="the Data Clerk returned no metrics for it"))
        return result
    result.gates.append(Gate(gate="no_metrics", passed=True))

    # freshness
    as_of = ((metrics.get("last_close") or {}).get("as_of"))
    if not as_of or value(metrics, "last_close") is None:
        result.gates.append(Gate(gate="freshness", passed=False, value=None, threshold=f"newest bar within {config.max_bar_age_sessions} session(s) of today", note=f"no bars: {note(metrics, 'last_close')}"))
    else:
        behind = sessions_between(date.fromisoformat(as_of), today)
        fundamentals = next((metrics[n]["as_of"] for n in ("roe", "pe_ratio") if (metrics.get(n) or {}).get("as_of")), None)
        old = fundamentals and (today - date.fromisoformat(fundamentals)).days > config.max_fundamentals_age_days
        reasons = [r for r, bad in ((f"the newest bar is {as_of}, {behind} sessions before {today}", behind > config.max_bar_age_sessions),
                                    (f"the Finnhub figures are from {fundamentals}", bool(old))) if bad]
        result.gates.append(Gate(gate="freshness", passed=not reasons, value={"newest_bar": as_of, "sessions_behind": behind, "fundamentals_as_of": fundamentals},
                                 threshold={"max_bar_age_sessions": config.max_bar_age_sessions, "max_fundamentals_age_days": config.max_fundamentals_age_days}, note="; ".join(reasons) or None))

    # red flags
    flags = [f"{e['class']} {e['age_bars']} bars ago" for e in events if e.get("class") in RED_FLAG_EVENTS and e.get("age_bars") is not None and e["age_bars"] <= config.red_flag_event_bars]
    debt, current = value(metrics, "debt_to_equity"), value(metrics, "current_ratio")
    waived = negative_equity(metrics)
    if debt is not None and current is not None and debt > config.red_flag_debt_to_equity and current < config.red_flag_current_ratio and not waived:
        flags.append(f"debt_to_equity {debt:g} with current_ratio {current:g}")
    if value(metrics, "cfo_negative_with_profit") is True:
        flags.append("cfo_negative_with_profit")
    result.gates.append(Gate(gate="red_flag", passed=not flags, value=flags, threshold="none", note=("; ".join([*flags, *([WAIVED_NOTE] if waived else [])]) or None)))

    # event risk
    days = value(metrics, "days_to_next_earnings")
    if days is not None:
        near = days <= config.earnings_days
        result.gates.append(Gate(gate="event_risk", passed=not near, value=days, threshold=f"more than {config.earnings_days} days to the next report", note=None))
    elif last_earnings_release:
        age = (today - date.fromisoformat(last_earnings_release)).days
        near = config.earnings_estimate_low <= age <= config.earnings_estimate_high
        result.gates.append(Gate(gate="event_risk", passed=not near, value=age, threshold=f"the last results release not {config.earnings_estimate_low} to {config.earnings_estimate_high} days old",
                                 note=f"estimated from the last earnings release ({last_earnings_release}, {age} days ago); the earnings calendar is not available"))
    else:
        result.gates.append(Gate(gate="event_risk", passed=True, value=None, note="no earnings date known: the calendar is not available and no results press release is stored"))
    return result
