"""Approved or denied (design choices entries 103 to 105; design/approved_or_denied_spec.md): the Critic's verdict is final, Auto mode asks nothing, and only the Account Manager touches the holdings.

REAL data only: the real reviews of the real session `tests/fixtures/session_real/bob_jones_oil.json` (XOM approved at 86, EOG approved at 76, APA denied for its risk) and of `pam_smith_neutral.json` (SNDK denied), and, for
Part C, the real Advisor captures. No stand-in agent and no stand-in model. The captures are not edited: a test that needs another shape of record makes it from the real one and says so.
"""
import copy
import json
import re
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest
from handshake_helpers import friday_close  # noqa: F401  (a fixture)

from prometheus.application import session
from prometheus.application.web import views
from prometheus.common import work_order as wo

SESSIONS = Path(__file__).parent / "fixtures" / "session_real"


def capture(name: str) -> dict:
    return json.loads((SESSIONS / f"{name}.json").read_text(encoding="utf-8"))


def reviewed(name: str = "bob_jones_oil") -> wo.WorkOrder:
    """The real work order as it was when the Critic had finished: before the Account Manager decided and before the Trader traded."""
    order = wo.WorkOrder.model_validate(copy.deepcopy(capture(name)["order"]))
    order.status, order.trade_decisions, order.trades, order.trader_runs = "reviewed", [], [], []
    return order


# ---------------------------------------------------------------- Part B: approved trades, nothing is asked in Auto mode


def test_a_run_where_everything_was_denied_says_so_in_its_first_reason_for_the_session_and_the_page():
    """The real SNDK review of `pam_smith_neutral.json` is the only review of that run and it was denied. Its trades were neutral purchases, so the order is made what a run with no neutral trade would leave: the Critic's
    review alone."""
    order = reviewed("pam_smith_neutral")
    order.trade_decisions = []
    assert len(order.reviews) == 1 and not order.reviews[0]["approved"]
    reasons = session.why_not(None, order)
    assert reasons[0] == "the Critic denied the recommendation and approved none: SNDK (risk_above_profile, falling_knife, dead_cat_bounce)"
    account = {"cash": Decimal("15000"), "positions": [], "invested": 0, "total": Decimal("15000")}
    verdict = views.run_verdict(account, account, action="buy", order=order)
    assert verdict["status"] == "no_trade" and verdict["headline"] == "Nothing was traded: " + reasons[0] and verdict["reasons"] == reasons


# ---------------------------------------------------------------- Part C: the Advisor and the Critic never touch the holdings (entry 105)

def keys_of(value):
    """Every key of every dict inside a record, however deep (a list is walked, a value is not a key)."""
    if isinstance(value, dict):
        for key, inner in value.items():
            yield key
            yield from keys_of(inner)
    elif isinstance(value, list):
        for inner in value:
            yield from keys_of(inner)


def model_keys(model, seen=None):
    """Every field name of a pydantic model and of the models inside it."""
    import typing

    from pydantic import BaseModel
    seen = seen if seen is not None else set()
    for name, field in model.model_fields.items():
        seen.add(name)
        for inner in typing.get_args(field.annotation) + (field.annotation,):
            args = typing.get_args(inner)
            for candidate in (inner, *args):
                if isinstance(candidate, type) and issubclass(candidate, BaseModel) and candidate is not model:
                    model_keys(candidate, seen)
    return seen


def test_a_version_2_record_loads_and_the_same_record_made_to_say_it_is_held_or_to_reduce_does_not():
    """The real version 2 recommendations of `bob_jones_oil.json` (every action `open`, nothing held) load, and come out as version 3 shape. The SAME record made to say something is held, or to reduce, add, hold or avoid, is
    refused with the reason; so is a record whose `portfolio_fit` carries a number, and a candidate that says it is held."""
    from pydantic import ValidationError

    from prometheus.agents.advisor.record import Candidate, Recommendation
    records = capture("bob_jones_oil")["order"]["recommendations"]
    assert len(records) == 3 and all(r["identity"]["schema_version"] == 2 and r["action"]["held_shares"] == 0 and "portfolio_fit" in r for r in records)
    for record in records:
        loaded = Recommendation.model_validate(record)
        assert loaded.action.action == "open" and loaded.concentration.note == record["portfolio_fit"]["note"] and "portfolio_fit" not in loaded.model_dump()
    record = copy.deepcopy(records[0])
    for change, message in (({"held_shares": 60}, "the record says something is held (held_shares 60)"), ({"cost_basis": 12.5}, "the record says something is held (cost_basis 12.5)"),
                            ({"unrealized_pct": 0.1}, "the record says something is held (unrealized_pct 0.1)")):
        with pytest.raises(ValidationError, match=re.escape(message)):
            Recommendation.model_validate({**record, "action": {**record["action"], **change}})
    for action in ("reduce", "add", "hold", "avoid"):
        with pytest.raises(ValidationError, match=f"the action is '{action}': a recommendation is to buy"):
            Recommendation.model_validate({**record, "action": {**record["action"], "action": action}})
    with pytest.raises(ValidationError, match="the record carries portfolio information"):
        Recommendation.model_validate({**record, "portfolio_fit": {**record["portfolio_fit"], "sector_weight_after": 0.2}})
    candidate = {"symbol": "ZZZ", "origin": "subject", "reason": "stance neutral", "held": False}
    assert Candidate.model_validate(candidate).symbol == "ZZZ" and "held" not in Candidate.model_validate(candidate).model_dump()
    with pytest.raises(ValidationError, match="the candidate says it is held"):
        Candidate.model_validate({**candidate, "held": True})


# ---------------------------------------------------------------- Part D: the money is spread round-robin in Ask mode too (entry 106)

def test_ask_mode_sizes_jacobs_request_as_auto_mode_did_with_the_same_real_prices_and_none_is_no_funds(friday_close):
    """The prices are the stored closes of Friday 2026-11-13, the day the two real sessions were priced at (`friday_close`): without the pin the desk reads the latest stored close, and the sizes move with the market.
    Jacob (2026-11-15), told that his Ask-mode request had put the whole $10,000 into F (839 shares) and left GM, PG and GOOGL with "no funds": *"No. its suppose to be round-robin. Not all in one."* The REAL desk over the
    same request and the same stored closes, in Ask mode: F 23, GM 19, PG 18 and GOOGL 17, exactly what the real Auto session did (`d_pyrett_names_auto.json`), and each is asked about and bought."""
    from test_direct_order import JacobDesk

    before = {d["symbol"]: (d["shares"], d["will_trade"]) for d in capture("d_pyrett_names_ask")["order"]["trade_decisions"]}
    assert before == {"F": (839, "yes_user"), "GM": (0, "no_funds"), "PG": (0, "no_funds"), "GOOGL": (0, "no_funds")}                       # what the real Ask session did: everything in the first stock
    auto = {d["symbol"]: d["shares"] for d in capture("d_pyrett_names_auto")["order"]["trade_decisions"]}
    assert auto == {"F": 23, "GM": 19, "PG": 18, "GOOGL": 17}
    run = JacobDesk("ask", answers=[True] * 5)
    order = run.state["order"]
    # AMENDED (entry 111, M3): a direct order with money buys as many whole shares as the money pays for and a FRACTION of a share with what is left (Jacob: "buy as many whole shares as possible and a fractional share for the leftover funds
    # available"): the leftover of the whole-share round robin goes to GOOGL, the stock with the fewest shares (the real Auto capture, made before, has 17); the other three are as they were
    assert {d["symbol"]: d["shares"] for d in order.trade_decisions} == {**auto, "GOOGL": Decimal("17.025628")} and not [d for d in order.trade_decisions if d["will_trade"] == "no_funds"]
    assert [(t["symbol"], t["status"], t["fill"]["shares"]) for t in order.trades] == [("F", "filled", 23), ("GM", "filled", 19), ("PG", "filled", 18), ("GOOGL", "filled", Decimal("17.025628"))]
    assert sum(d["est_cost"] for d in order.trade_decisions) <= Decimal("10000")
