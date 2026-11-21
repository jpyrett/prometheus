"""The Advisor's trade plan, on the REAL captured metrics, before and after the size left it (design choices entry 80).

`tests/fixtures/advisor_real/plans_before_size_removal.json` holds every NON-SIZE number of the plans the Advisor's code made BEFORE the change: 208 plans (a plan for each of the six strategy families and for a
reduction, with and without a quote, and the two reactions of an event) over every symbol the three real captures (`SNDK`, `PG`, `PG_open`) hold metrics for: the real ATRs, 50-day means, closes, spreads and
earnings fields the Data Clerk calculated on 2026-11-05. It was made ONCE by running the old `trade_plan.build` on those metrics (with the capture's real portfolio, so that the old code sized every plan) and keeping
every field except the nine size fields it removed. The old code's plans did not depend on the regime apart from the size (checked when the file was made: the trend and the stress regime gave the same numbers).

What these tests prove: removing the risk budget, the position and liquidity caps, the equity and the funds check from `build` changed NO price, no date, no note and no rule of the plan: the entry zone, the entry
mid, the stop distance and price, the target, the time stop, the order style, the limit price, the invalidation rules, the gap risk and the notes are the same for every one of the 208. The only note that went is the
one that said the size could not be worked out ("no size: equity is unknown"); no real capture had it. Nothing here is invented: the quote, when there is one, is the symbol's own last close.

AMENDED (Advisor config version 3, design/advisor_exit_plan_spec.md rules E1 and E2; design choices entry 139): the file is the plans of Advisor config version 2 and is NOT edited. Version 3 changes ONE thing about a plan: its TARGET and the trailing stop that goes with
it. A target is at least 2 x the stop distance above the entry mid (conviction, event, reversion and theme: the edge's target when it is already higher), the TREND family has no target and a stop that trails the highest close (`trailing`, `trail_distance` = the stop distance), and
capital_return keeps no target. The tests below compare all 182 plans with the old file and say exactly what is allowed to differ.

AMENDED (design choices entry 105): the file's 26 plans for a `reduce` (a plan with an entry zone and nothing else, the family None) are the plans of a sale the Advisor no longer recommends: `build` needs a family now, and
those 26 are not compared. The other 182 are, unchanged. The file is a capture and is not edited.
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
def everything():
    return json.loads((FIXTURES / "plans_before_size_removal.json").read_text(encoding="utf-8"))["plans"]


@pytest.fixture(scope="module")
def golden(everything):
    """The 182 plans of a purchase: the file less its 26 plans for a `reduce` (entry 105: the Advisor recommends no sale)."""
    return {k: v for k, v in everything.items() if k.split("|")[2] != "reduce"}


@pytest.fixture(scope="module")
def captures():
    return {name: json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8")) for name in ("SNDK", "PG", "PG_open")}


def family_of(name: str) -> FamilyResult:
    """The family the file was made with: its own horizon from the settings and its own prior as the edge (the same inputs the old code was given)."""
    c = getattr(CONFIG.families, name)
    prior = getattr(c, "prior_20d", None) or getattr(c, "prior_event", 0.008)
    return FamilyResult(family=name, computable=True, passed=True, direction="bullish", horizon_bars=c.horizon, prior_20d=prior, edge_20d=prior)


def new_plan(key: str, captures: dict):
    name, symbol, family, quote, reaction = key.split("|")
    fx = captures[name]
    metrics = fx["metrics"][symbol]
    last = (metrics.get("last_close") or {}).get("value")
    return tp.build(family_of(family), metrics, {"price": last} if quote == "quote_at_last_close" and last else None, date.fromisoformat(fx["today"]), CONFIG, symbol, reaction_atr=None if reaction == "None" else float(reaction))


def test_the_file_covers_every_family_both_entry_prices_and_the_rules_that_change_a_plan(everything, golden):
    keys = [k.split("|") for k in everything]
    assert len(everything) == 208 and len(golden) == 182 and sum(1 for k in keys if k[2] == "reduce") == 26 and {k[2] for k in keys} == {*FAMILIES, "reduce"} and {k[3] for k in keys} == {"no_quote", "quote_at_last_close"} and {k[0] for k in keys} == {"SNDK", "PG", "PG_open"}
    assert {k[1] for k in keys} >= {"SNDK", "PG", "MSFT", "TSLA", "GM"} and {v["entry_basis"] for v in golden.values()} == {"last_close", "quote", "none"}
    assert {v["order_style"] for v in golden.values()} == {None, "limit", "marketable_limit"} and any(v["target_price"] is None and v["stop_price"] for v in golden.values())
    assert any("the stop is the 50-day mean" in n for v in golden.values() for n in v["notes"]) and any("one ATR or more" in n for v in golden.values() for n in v["notes"])
    assert not any(GONE & set(v) for v in golden.values())                                                          # the file holds no size


EXIT_NOTES = ("the target is ", "trailing stop, no target")      # the two sentences Advisor config version 3 adds to a plan's notes


def test_every_number_of_the_plan_is_what_the_old_code_made_for_all_182_real_plans_but_the_target_and_the_trail(golden, captures):
    """Every field of the 182 plans is the old one except the target and the two trailing fields (and a note that says why): the entry zone, mid, stop distance and price, time stop, order style, limit price, invalidation, gap risk and the other notes are unchanged.
    The target: capital_return and a plan with no prices keep theirs; the trend family has none and trails (`trailing` true, `trail_distance` the stop distance); every other family has the old target or, where that is nearer than 2 x the stop distance above the entry
    mid, the entry mid plus 2 x the stop distance (worked here by hand from the OLD file's own entry mid and stop distance, to the 4 places the plan is written in)."""
    different, raised, trailed = [], 0, 0
    for key, golden_plan in golden.items():
        old = dict(golden_plan)                                                  # the module's file is read once: never changed
        new = new_plan(key, captures).model_dump(mode="json")
        family = key.split("|")[2]
        notes = [n for n in new.pop("notes") if not n.startswith(EXIT_NOTES)]
        trailing, trail, target = new.pop("trailing"), new.pop("trail_distance"), new.pop("target_price")
        old_target, old_notes = old.pop("target_price"), old.pop("notes")
        if new != old or notes != old_notes:
            different.append((key, {k: (old.get(k), new.get(k)) for k in {*old, *new} if old.get(k) != new.get(k)}))
        priced = old["stop_price"] is not None
        assert trailing is (family == "trend" and priced) and trail == (old["stop_distance"] if trailing else None), key
        if not priced or family == "capital_return":
            assert target == old_target, key                                      # no prices, or bought for the payout: no target, as before
        elif trailing:
            assert target is None, key                                            # the trend family runs its winners: no target
            trailed += 1
        else:
            floor = round(old["entry_mid"] + 2 * old["stop_distance"], 4)
            assert old_target is not None and abs(target - max(old_target, floor)) <= 0.0002 and target - old["entry_mid"] >= 2 * old["stop_distance"] - 0.0002, key
            raised += target > old_target + 0.0002
    assert different == [] and raised > 0 and trailed > 0         # the 2 for 1 floor moved targets and the trailing stop took the trend family's


def test_the_new_plans_carry_exactly_the_fields_the_old_ones_had_less_the_size_and_the_two_trailing_fields(golden, captures):
    for key in golden:
        assert set(new_plan(key, captures).model_dump(mode="json")) == set(next(iter(golden.values()))) | {"trailing", "trail_distance"}


def test_there_is_no_plan_for_a_reduction_any_more_build_needs_the_strategy_a_purchase_is_made_on():
    import inspect
    assert "None" not in str(inspect.signature(tp.build).parameters["family"].annotation)
    with pytest.raises(AttributeError):                                                  # a plan with no family (the old `reduce` plan) cannot be built
        tp.build(None, {"last_close": {"value": 10.0}, "atr_14": {"value": 1.0}}, None, date(2026, 11, 5), CONFIG, "X")


def test_a_real_plan_checked_by_hand(captures):
    """SNDK on capital_return, from its real numbers: last close 806.43, ATR 108.6778; entry zone last +- 0.5 ATR, stop 3 ATR below the entry, no target, a limit at the entry mid, 126 weekdays after 2026-11-05."""
    metrics = captures["SNDK"]["metrics"]["SNDK"]
    atr = metrics["atr_14"]["value"]
    plan = new_plan("SNDK|SNDK|capital_return|no_quote|None", captures)
    assert plan.entry_mid == 806.43 and plan.entry_zone == [round(806.43 - 0.5 * atr, 4), round(806.43 + 0.5 * atr, 4)] and plan.stop_distance == round(3 * atr, 4) and plan.stop_price == round(806.43 - 3 * atr, 4)
    assert plan.target_price is None and plan.order_style == "limit" and plan.limit_price == 806.43 and plan.time_stop_date == "2027-04-30" and plan.horizon_bars == 126
    assert [r["kind"] for r in plan.invalidation] == ["price_below", "days_elapsed", "event_class"]


def test_the_real_trend_plan_of_procter_and_gamble_trails_and_its_conviction_plan_pays_two_for_one(captures):
    """PG (Advisor capture PG_open, 2026-11-05) from its real numbers, by hand: the trend plan has no target and trails by its stop distance; the conviction plan's target is the entry mid plus 2 x its stop distance, because the family's edge target (0.6% a month,
    40 sessions: 1.2%) is nearer than that."""
    metrics = captures["PG_open"]["metrics"]["PG"]
    last, atr = metrics["last_close"]["value"], metrics["atr_14"]["value"]
    trend = new_plan("PG_open|PG|trend|no_quote|None", captures)
    assert trend.entry_mid == round(last, 4) and trend.target_price is None and trend.trailing is True and trend.trail_distance == trend.stop_distance and trend.stop_price == round(last - trend.stop_distance, 4)
    assert trend.stop_distance == round(3 * atr, 4) or any("the stop is the 50-day mean" in n for n in trend.notes)       # 3 ATR, or the distance to the 50-day mean when that is nearer
    conviction = new_plan("PG_open|PG|conviction|no_quote|None", captures)
    assert conviction.trailing is False and conviction.target_price == round(max(last * (1 + 1.5 * 0.012), last + 2 * conviction.stop_distance), 4) == round(last + 2 * conviction.stop_distance, 4)
    assert conviction.target_price > round(last * 1.018, 4) and any("the family's edge alone gave" in n for n in conviction.notes)


def test_the_report_says_a_trend_plan_trails_and_a_capital_return_plan_is_held_for_the_payout(captures):
    """The runner's report (`recommendations.plan_target`) no longer says "held for the payout" of every plan without a target: only SNDK's capital-return plan is; PG's trend plan says how far its stop trails."""
    from prometheus.application import recommendations as report
    trend = new_plan("PG_open|PG|trend|no_quote|None", captures)
    assert report.plan_target(trend.model_dump()) == f"none (the stop trails the highest close by ${trend.trail_distance:,.2f})"
    assert report.plan_target(new_plan("SNDK|SNDK|capital_return|no_quote|None", captures).model_dump()) == "none (held for the payout)"
    assert report.plan_target(new_plan("PG_open|PG|conviction|no_quote|None", captures).model_dump()) == f"${new_plan('PG_open|PG|conviction|no_quote|None', captures).target_price:,.2f}"


def test_the_order_in_the_captures_is_data_ready_and_the_metrics_are_the_real_ones(captures):
    for name, fx in captures.items():
        assert wo.WorkOrder.model_validate(fx["order"]).status == "data_ready" and fx["today"] == "2026-11-05" and fx["metrics"][fx["symbol"]]["last_close"]["as_of"] == "2026-11-05"
    assert (REPO_ROOT / captures["SNDK"]["portfolio"]).exists()
