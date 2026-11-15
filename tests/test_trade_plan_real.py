"""The Advisor's trade plan, on the REAL captured metrics, before and after the size left it (design choices entry 80).

`tests/fixtures/advisor_real/plans_before_size_removal.json` holds every NON-SIZE number of the plans the Advisor's code made BEFORE the change: 208 plans (a plan for each of the six strategy families and for a
reduction, with and without a quote, and the two reactions of an event) over every symbol the three real captures (`SNDK`, `PG`, `PG_open`) hold metrics for: the real ATRs, 50-day means, closes, spreads and
earnings fields the Data Clerk calculated on 2026-11-05. It was made ONCE by running the old `trade_plan.build` on those metrics (with the capture's real portfolio, so that the old code sized every plan) and keeping
every field except the nine size fields it removed. The old code's plans did not depend on the regime apart from the size (checked when the file was made: the trend and the stress regime gave the same numbers).

What these tests prove: removing the risk budget, the position and liquidity caps, the equity and the funds check from `build` changed NO price, no date, no note and no rule of the plan: the entry zone, the entry
mid, the stop distance and price, the target, the time stop, the order style, the limit price, the invalidation rules, the gap risk and the notes are the same for every one of the 208. The only note that went is the
one that said the size could not be worked out ("no size: equity is unknown"); no real capture had it. Nothing here is invented: the quote, when there is one, is the symbol's own last close.
"""
import json
from datetime import date
from pathlib import Path

import pytest
from advisor_helpers import CONFIG

from prometheus.agents.advisor import trade_plan as tp
from prometheus.agents.advisor.record import FamilyResult
from prometheus.common import work_order as wo
from prometheus.rag.config import REPO_ROOT

FIXTURES = Path(__file__).parent / "fixtures" / "advisor_real"
GONE = {"equity", "equity_basis", "risk_budget", "suggested_shares", "position_pct", "oversized", "size_note", "funds_feasible", "funds_note"}
FAMILIES = ("trend", "event", "reversion", "capital_return", "theme", "conviction")


@pytest.fixture(scope="module")
def golden():
    return json.loads((FIXTURES / "plans_before_size_removal.json").read_text(encoding="utf-8"))["plans"]


@pytest.fixture(scope="module")
def captures():
    return {name: json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8")) for name in ("SNDK", "PG", "PG_open")}


def family_of(name: str) -> FamilyResult | None:
    """The family the file was made with: its own horizon from the settings and its own prior as the edge (the same inputs the old code was given)."""
    if name == "reduce":
        return None
    c = getattr(CONFIG.families, name)
    prior = getattr(c, "prior_20d", None) or getattr(c, "prior_event", 0.008)
    return FamilyResult(family=name, computable=True, passed=True, direction="bullish", horizon_bars=c.horizon, prior_20d=prior, edge_20d=prior)


def new_plan(key: str, captures: dict):
    name, symbol, family, quote, reaction = key.split("|")
    fx = captures[name]
    metrics = fx["metrics"][symbol]
    last = (metrics.get("last_close") or {}).get("value")
    return tp.build(family_of(family), metrics, {"price": last} if quote == "quote_at_last_close" and last else None, date.fromisoformat(fx["today"]), CONFIG, symbol, reaction_atr=None if reaction == "None" else float(reaction))


def test_the_file_covers_every_family_both_entry_prices_and_the_rules_that_change_a_plan(golden):
    keys = [k.split("|") for k in golden]
    assert len(golden) == 208 and {k[2] for k in keys} == {*FAMILIES, "reduce"} and {k[3] for k in keys} == {"no_quote", "quote_at_last_close"} and {k[0] for k in keys} == {"SNDK", "PG", "PG_open"}
    assert {k[1] for k in keys} >= {"SNDK", "PG", "MSFT", "TSLA", "GM"} and {v["entry_basis"] for v in golden.values()} == {"last_close", "quote", "none"}
    assert {v["order_style"] for v in golden.values()} == {None, "limit", "marketable_limit"} and any(v["target_price"] is None and v["stop_price"] for v in golden.values())
    assert any("the stop is the 50-day mean" in n for v in golden.values() for n in v["notes"]) and any("one ATR or more" in n for v in golden.values() for n in v["notes"])
    assert not any(GONE & set(v) for v in golden.values())                                                          # the file holds no size


def test_every_non_size_number_of_the_plan_is_what_the_old_code_made_for_all_208_real_plans(golden, captures):
    different = []
    for key, old in golden.items():
        new = new_plan(key, captures).model_dump(mode="json")
        if new != old:
            different.append((key, {k: (old.get(k), new.get(k)) for k in {*old, *new} if old.get(k) != new.get(k)}))
    assert different == []


def test_the_new_plans_carry_exactly_the_fields_the_old_ones_had_less_the_size(golden, captures):
    for key in golden:
        assert set(new_plan(key, captures).model_dump(mode="json")) == set(next(iter(golden.values())))


def test_a_real_plan_checked_by_hand(captures):
    """SNDK on capital_return, from its real numbers: last close 806.43, ATR 108.6778; entry zone last +- 0.5 ATR, stop 3 ATR below the entry, no target, a limit at the entry mid, 126 weekdays after 2026-11-05."""
    metrics = captures["SNDK"]["metrics"]["SNDK"]
    atr = metrics["atr_14"]["value"]
    plan = new_plan("SNDK|SNDK|capital_return|no_quote|None", captures)
    assert plan.entry_mid == 806.43 and plan.entry_zone == [round(806.43 - 0.5 * atr, 4), round(806.43 + 0.5 * atr, 4)] and plan.stop_distance == round(3 * atr, 4) and plan.stop_price == round(806.43 - 3 * atr, 4)
    assert plan.target_price is None and plan.order_style == "limit" and plan.limit_price == 806.43 and plan.time_stop_date == "2027-04-30" and plan.horizon_bars == 126
    assert [r["kind"] for r in plan.invalidation] == ["price_below", "days_elapsed", "event_class"]


def test_the_order_in_the_captures_is_data_ready_and_the_metrics_are_the_real_ones(captures):
    for name, fx in captures.items():
        assert wo.WorkOrder.model_validate(fx["order"]).status == "data_ready" and fx["today"] == "2026-11-05" and fx["metrics"][fx["symbol"]]["last_close"]["as_of"] == "2026-11-05"
    assert (REPO_ROOT / captures["SNDK"]["portfolio"]).exists()
