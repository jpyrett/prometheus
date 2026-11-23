"""A PRICE FLOOR AND A TYPED NUMBER OF SHARES WITH A RECOMMENDATION (design choices entry 140; design/limit_orders_spec.md, parts A and C; Jacob, 2026-11-21).

REAL DATA ONLY, as `test_request_numbers.py`, whose helpers these tests share. The Account Manager's model is a REPLAY of its REAL replies recorded on 2026-11-20 with the prompt of entry 140 (the form has `min_price`):
`tests/fixtures/session_real/request_numbers_forms.json`, the eight requests that follow the first twenty-one ("Buy the top 3 AI stocks over $20 a share", "Sell 10 shares of MSFT at $600 or more", "Buy 10 of MU at $90
or less", "buy 5 shares of each of the top 3 oil stocks if recommended", ... and "Buy 10 of MU if the recommendation passes" recorded AGAIN under the key `<text> [entry 140]`, next to the reply of the earlier prompt). The real code
checks every field of those replies against the words the user typed. The decisions are the real Account Manager's over the real work order of `bob_jones_oil.json` (XOM $170.59 and EOG $158.32 approved, APA denied) and the
real neutral session `pam_smith_neutral.json` (MU $778.31, AMD $534.06), the prices the stored closes of Friday 2026-11-13, the Trader the real Trader with no model. A form a test writes itself ("a model that said ...") has the
shape of the real replies and is for a FAULT. Nothing else stands in for an agent, a model or a market number.
"""
import copy
import json
from datetime import date
from decimal import Decimal

import pytest
from handshake_helpers import BOB, BUDGET, MOMENT, PROFILE, close_of, friday_close, jacob_as_bob, manager, trader  # noqa: F401  (friday_close is a fixture)
from test_account_manager_clarify import channel_of
from test_direct_order import clerk, never, Replay
from test_neutral_recommendation import neutral_only
from test_proving_spare import no_model, put_research, walk  # noqa: F401  (no_model is a fixture)
from test_request_numbers import FORMS, ADVISOR_FIXTURES, ASK, AUTO, MSFT_SKIP, fields, form, limited, order_of, read, real, replay, reviewed  # noqa: F401

from prometheus.agents.account_manager import decisions, handshake
from prometheus.agents.account_manager import intent as it
from prometheus.agents.account_manager.account_manager import AccountManager
from prometheus.agents.advisor.recommend import within_price_limit
from prometheus.common import work_order as wo
from prometheus.common.money import dumps
from prometheus.common.portfolio import Portfolio
from prometheus.common.user_profile import UserProfile
from prometheus.langgraph import desk

pytestmark = pytest.mark.usefixtures("friday_close")

OVER_20 = "Buy the top 3 AI stocks over $20 a share"
BETWEEN = "Recommend 3 oil stocks between $20 and $60 a share"
SELL_600 = "Sell 10 shares of MSFT at $600 or more"
MU_90 = "Buy 10 of MU at $90 or less"
MSFT_OVER_20 = "buy MSFT over $20 a share"
EACH_5 = "buy 5 shares of each of the top 3 oil stocks if recommended"
MU_AMD = "Buy 10 of MU and 5 of AMD if they're recommended"
MU_PASSES = "Buy 10 of MU if the recommendation passes"
MU_PASSES_NEW = MU_PASSES + " [entry 140]"


def reads(key: str) -> it.TradeIntent:
    """What the code makes of the model's last real reply to the request recorded under `key` (the text the user typed is the key, less the mark of a request recorded again)."""
    return it.parse_intent(real(key), key.removesuffix(" [entry 140]"))


# ---------------------------------------------------------------- the recordings are what they say they are

def test_the_recordings_of_entry_140_are_the_real_models_replies_and_the_older_ones_are_as_they_were():
    assert len(FORMS) == 33 and {OVER_20, BETWEEN, SELL_600, MU_90, MSFT_OVER_20, EACH_5, MU_AMD, MU_PASSES, MU_PASSES_NEW} <= set(FORMS)      # AMENDED (entry 146): four more recordings (was 29)
    assert json.loads(real(OVER_20))["min_price"] == 20 and json.loads(real(BETWEEN))["min_price"] == 20 and json.loads(real(BETWEEN))["max_price"] == 60
    assert json.loads(real(SELL_600))["orders"] == [{"side": "sell", "symbol": "MSFT", "shares": 10, "dollars": None}] and json.loads(real(MU_90))["max_price"] == 90
    assert "min_price" not in json.loads(real(MU_PASSES)) and json.loads(real(MU_PASSES_NEW))["orders"] == [] and len(FORMS[MU_PASSES_NEW]["replies"]) == 2      # the new prompt: "buy", no orders; the first reply named no stock and code refused it


# ---------------------------------------------------------------- F1: a floor is a $ number tied to one share, written as a lower bound

@pytest.mark.parametrize("typed, kinds", [
    (OVER_20, [("floor", 20)]), (BETWEEN, [("floor", 20), ("ceiling", 60)]), (SELL_600, [("floor", 600)]), (MU_90, [("ceiling", 90)]), (MSFT_OVER_20, [("floor", 20)]),
    ("sell at $600 or higher", [("floor", 600)]), ("sell at $600 or better", [("floor", 600)]), ("buy MU at $90 or better", [("ceiling", 90)]),      # "or better" is the side's: a sale wants more, a purchase less
    ("recommend stocks at $90 or better", [("ceiling", 90)]),      # a recommendation is a purchase
    ("buy MU at $90 or lower", [("ceiling", 90)]), ("stocks priced over $100", [("floor", 100)]), ("at least $20 a share", [("floor", 20)]), ("no less than $20 a share", [("floor", 20)]),
    ("sell at $20 a share or more", [("floor", 20)]), ("buy MU between $20 and $60", [("money", 20), ("money", 60)]),      # no share beside either: an amount to spend, as before
    ("between $60 and $20 a share", [("ceiling", 60), ("floor", 20)]),      # the lower is the floor
    ("buy MU if the price is $90 or less", [("ceiling", 90)]), ("with $1,000 or more", [("money", 1000)]), ("buy the top 3 with $1,000 or less", [("money", 1000)]), ("a budget of $2,000 or less", [("money", 2000)]),
    ("buy $5,000 or more of MU", [("money", 5000)]), ("invest $5,000 or less", [("money", 5000)]), ("spend $600 or more on MU", [("money", 600)]),      # an amount to spend, not a price
    ("spend between $1000 and $2000 on MU", [("money", 1000), ("money", 2000)]), ("buy between $20 and $60 of MU", [("money", 20), ("money", 60)]),
])
def test_a_floor_is_a_lower_bound_tied_to_a_share_and_never_an_amount_to_spend(typed, kinds):
    assert [(w.kind, w.value) for w in it.written_amounts(typed)] == [(k, Decimal(v)) for k, v in kinds]


def test_a_floor_is_never_money_to_spend_and_is_no_answer_to_how_much():
    assert it.money_amounts(OVER_20) == [] and it.money_amounts(SELL_600) == [] and it.money_amounts(BETWEEN) == [] and it.price_floors(BETWEEN) == [Decimal(20)] and it.price_limits(BETWEEN) == [Decimal(60)]
    assert reads(OVER_20).said_how_much() is False and reads(BETWEEN).said_how_much() is False


def test_the_real_reply_to_a_theme_with_a_floor_carries_it_with_the_count_and_the_theme_and_says_nothing_to_correct():
    intent = reads(OVER_20)
    assert (intent.action, intent.topic, intent.count, intent.min_price, intent.max_price, intent.budget) == ("buy", "AI", 3, Decimal(20), None, None) and intent.corrected == []
    assert intent.describe().endswith("only while a share costs $20.00 or more") and type(intent.min_price) is Decimal
    assert json.loads(dumps(intent.model_dump()))["min_price"] == 20 and it.TradeIntent.model_validate(intent.model_dump()).min_price == Decimal(20)      # a number in JSON, read back


def test_the_real_reply_to_a_range_is_a_look_with_both_bounds():
    intent = reads(BETWEEN)
    assert (intent.action, intent.topic, intent.count, intent.min_price, intent.max_price) == ("research", "oil", 3, Decimal(20), Decimal(60)) and "nothing will be bought" in intent.corrected[0]
    assert "only while a share costs between $20.00 and $60.00" in intent.describe()


def test_the_real_reply_to_a_sale_at_a_price_or_more_is_a_sale_with_a_floor_and_no_ceiling():
    intent = reads(SELL_600)
    assert intent.action == "order" and fields(intent) == [("sell", "MSFT", Decimal(10), None)] and intent.min_price == Decimal(600) and intent.max_price is None and [o.min_price for o in intent.orders] == [Decimal(600)]
    assert intent.corrected == [] and intent.describe().startswith("sell 10 shares of MSFT, only while a share costs $600.00 or more")


def test_the_real_reply_to_a_purchase_at_a_price_or_less_is_a_ceiling_and_the_real_reply_to_a_floor_on_a_purchase_is_a_floor_on_that_order():
    ceiling = reads(MU_90)
    assert ceiling.max_price == Decimal(90) and ceiling.min_price is None and [(o.max_price, o.min_price) for o in ceiling.orders] == [(Decimal(90), None)] and fields(ceiling) == [("buy", "MU", Decimal(10), None)]
    floor = reads(MSFT_OVER_20)
    assert floor.action == "order" and fields(floor) == [("buy", "MSFT", None, None)] and floor.min_price == Decimal(20) and [o.min_price for o in floor.orders] == [Decimal(20)]
    dump = floor.orders[0].model_dump()
    assert dump["min_price"] == Decimal(20) and "min_price" not in it.DirectOrder(side="buy", symbol="MU", shares=1).model_dump() and "max_price" not in dump      # a dump has the key only when the user gave one


def test_a_floor_the_model_missed_is_taken_from_the_words_one_it_made_up_is_refused_and_the_words_win():
    missed = it.parse_intent(form(orders=[{"side": "sell", "symbol": "MSFT", "shares": 10, "dollars": None}]), SELL_600)
    assert missed.min_price == Decimal(600) and "your minimum price" in missed.corrected[0]
    wrong = it.parse_intent(form(orders=[{"side": "sell", "symbol": "MSFT", "shares": 10, "dollars": None}], min_price=500), SELL_600)
    assert wrong.min_price == Decimal(600)
    with pytest.raises(ValueError, match="is not a minimum price that is in what you typed"):
        it.parse_intent(form(orders=[order_of("MU", 10)], min_price=50), "Buy 10 of MU")
    with pytest.raises(ValueError, match="min_price must be a number"):
        it.parse_intent(form(orders=[order_of("MU", 10)], min_price="50"), "Buy 10 of MU over $50 a share")
    with pytest.raises(ValueError, match="at most 4 decimals"):
        it.parse_intent(form(orders=[order_of("MU", 10)]), "Buy 10 of MU over $50.12345 a share")
    on_the_order = it.parse_intent(form(orders=[{**order_of("MU", 10), "min_price": 5, "max_price": 80}]), "Buy 10 of MU")      # the request's, read from the words: what the model wrote on an order is not taken
    assert on_the_order.min_price is None and on_the_order.max_price is None and [(o.min_price, o.max_price) for o in on_the_order.orders] == [(None, None)]


def test_the_floor_the_model_read_as_the_money_to_spend_is_taken_back_and_two_floors_or_a_floor_above_the_ceiling_are_for_the_user_to_clear_up():
    intent = it.parse_intent(form(orders=[order_of("MU", None, 20)]), "Buy MU over $20 a share")
    assert fields(intent) == [("buy", "MU", None, None)] and intent.min_price == Decimal(20) and any("price per share (your limit)" in c for c in intent.corrected)
    theme = it.parse_intent(form(action="buy", topic="oil", count=2, budget=20), "Buy 2 oil stocks over $20 a share")
    assert theme.budget is None and theme.min_price == Decimal(20)
    with pytest.raises(it.NeedsClarification, match="more than one minimum price"):
        it.parse_intent(form(action="buy", topic="oil", count=2), "Buy 2 oil stocks over $20 a share and 2 gas ones over $30 a share")
    with pytest.raises(it.NeedsClarification, match="above your price limit"):
        it.parse_intent(form(action="buy", topic="oil", count=2, max_price=20), "Buy 2 oil stocks over $60 a share and under $20 a share")
    mixed = form(orders=[order_of("MU", 10), {"side": "sell", "symbol": "MSFT", "shares": 10, "dollars": None}])
    with pytest.raises(it.NeedsClarification, match="a purchase and a sale"):
        it.parse_intent(mixed, "Buy 10 of MU and sell 10 MSFT at $600 or more")      # a floor goes with ONE trade: a purchase that waits for a rise is not built
    assert it.parse_intent(mixed, "Buy 10 of MU and sell 10 MSFT").min_price is None


def test_the_prompt_asks_for_a_floor_by_its_own_name_says_a_range_and_says_who_reads_the_shares_typed_with_a_recommendation():
    from prometheus.prompts import account_manager_request as prompt
    assert '"min_price": null' in prompt.SYSTEM and "PRICE FLOOR" in prompt.SYSTEM and "between $20 and $60 a share" in prompt.SYSTEM and "a program reads the shares from the person's words" in prompt.SYSTEM and "min_price" in prompt.RETRY
    assert "is not supported" not in prompt.SYSTEM and not any(real_name in prompt.SYSTEM for real_name in ("MU", "KO", "AMD", "MSFT", "Micron", "XOM"))


# ---------------------------------------------------------------- F2: the Advisor drops what costs less, the Account Manager gives no decision, the handshake refuses a fill under it

def test_the_advisor_s_filter_is_one_function_for_both_bounds_and_the_floor_leaves_out_the_cheaper_ones_with_their_reason():
    capture = json.loads((ADVISOR_FIXTURES.parent / "session_real" / "auto_ai_quoted_theme.json").read_text(encoding="utf-8"))["order"]
    from advisor_helpers import scored
    ranked = [scored(symbol=r["symbol"], metrics={name: {**value, "window": "test", "inputs_summary": {}, "note": None} for name, value in r["features"].items()}, rank=False) for r in capture["recommendations"]]
    assert {s.symbol: s.metrics["last_close"]["value"] for s in ranked} == {"MSFT": 523.88, "AMD": 534.06, "MU": 778.31}
    why: dict = {}
    assert [s.symbol for s in within_price_limit(ranked, {}, None, why, Decimal("530"))] == ["AMD", "MU"] and why == {"MSFT": {"reason": "price $523.88 is under your minimum of $530.00 a share"}}
    why = {}
    assert [s.symbol for s in within_price_limit(ranked, {}, Decimal("600"), why, Decimal("530"))] == ["AMD"] and set(why) == {"MSFT", "MU"}      # a range: AMD is the one inside it
    assert why["MU"]["reason"] == "price $778.31 is above your limit of $600.00 a share" and why["MSFT"]["reason"] == "price $523.88 is under your minimum of $530.00 a share"
    assert within_price_limit(ranked, {}, None, {}, None) is ranked and within_price_limit(ranked, {}, None, {}) is ranked      # neither bound: the same list
    assert [s.symbol for s in within_price_limit(ranked, {}, None, {}, Decimal("523.88"))] == ["MSFT", "AMD", "MU"]      # AT the floor is inside it
    assert within_price_limit(ranked[:1], {"MSFT": {"price": 520.0}}, None, why := {}, Decimal("523")) == [] and why["MSFT"]["reason"] == "price $520.00 is under your minimum of $523.00 a share"      # the order's quote is the price the plan uses


def test_a_candidate_with_no_price_cannot_be_shown_to_be_over_the_floor_and_is_left_out():
    from advisor_helpers import scored
    nothing = scored(symbol="ZZZ", metrics={"last_close": {"value": None, "unit": "usd", "as_of": "2026-10-23", "window": "test", "source": "test", "inputs_summary": {}, "note": "needs bars"}}, rank=False)
    why: dict = {}
    assert within_price_limit([nothing], {}, None, why, Decimal(50)) == [] and why["ZZZ"]["reason"] == "no price to check against your minimum of $50.00 a share"


def test_the_real_advisor_step_recommends_a_stock_over_the_floor_and_leaves_one_under_it_out_with_its_reason():
    """The real Advisor step on the real captured inputs of `tests/fixtures/advisor_real/SNDK.json` (its model a replay of its real replies), with the user's price FLOOR on the order (SNDK closed at $806.43)."""
    from advisor_helpers import MemoryLedger, Scripted
    from test_advisor_sndk_pg import RecordedClerk

    from prometheus.agents.advisor.advisor import Advisor

    def step(floor):
        fx = json.loads((ADVISOR_FIXTURES / "SNDK.json").read_text(encoding="utf-8"))
        order = wo.WorkOrder.model_validate(fx["order"])
        order.min_price = floor
        Advisor(model=Scripted(*fx["replies"]), clerk=RecordedClerk(fx["metrics"]), ledger=MemoryLedger()).generate_recommendations(order, "funds", date.fromisoformat(fx["today"]))
        return order

    assert [r["symbol"] for r in step(None).recommendations] == ["SNDK"]
    assert [r["symbol"] for r in step(Decimal("800")).recommendations] == ["SNDK"]      # $806.43 is over $800
    under = step(Decimal("900"))
    assert under.recommendations == [] and under.status == "recommended" and under.errors == []
    (left,) = [c for c in under.candidates if c["symbol"] == "SNDK"]
    assert left["reason"] == "price $806.43 is under your minimum of $900.00 a share"
    assert [r["symbol"] for r in step(Decimal("806.43")).recommendations] == ["SNDK"]      # AT the floor is over it


def floored(floor, limit=None, budget=BUDGET):
    """The real session of `bob_jones_oil.json` decided again with the user's price floor (and limit)."""
    order = reviewed()
    return order, decisions.decide_trades(order, BOB, PROFILE, None, MOMENT, "auto", "even", budget, max_price=limit, min_price=floor)


def test_an_approved_purchase_priced_under_the_floor_gets_no_decision_and_the_others_carry_the_floor():
    assert all("min_price" not in d for d in floored(None)[1].trade_decisions)
    order = floored(Decimal("160"))[1]
    assert [(d["symbol"], d["will_trade"]) for d in order.trade_decisions] == [("XOM", "yes_auto"), ("APA", "no_system_denied")]      # EOG ($158.32) has none; APA was denied by the Critic, not by the floor
    assert order.trade_decisions[0]["min_price"] == Decimal("160.0000") and "min_price" not in order.trade_decisions[1] and order.trade_decisions[0]["shares"] > 0 and order.trade_decisions[0]["size"]["pot"] == BUDGET
    assert [p.symbol for p in decisions.purchases_to_size(reviewed(), BOB, MOMENT, None, Decimal("160"))] == ["XOM"] and [p.symbol for p in decisions.purchases_to_size(reviewed(), BOB, MOMENT, Decimal("165"), Decimal("160"))] == []
    assert decisions.has_purchase_to_size(reviewed(), MOMENT, None, Decimal("300")) is False and decisions.has_purchase_to_size(reviewed(), MOMENT, None, Decimal("160")) is True
    nothing = floored(Decimal("300"))[1]
    assert [(d["symbol"], d["will_trade"]) for d in nothing.trade_decisions] == [("APA", "no_system_denied")] and nothing.status == "decided"
    assert [d["symbol"] for d in floored(Decimal("158.32"))[1].trade_decisions] == ["XOM", "APA", "EOG"]      # AT the floor is over it
    both = floored(Decimal("160"), Decimal("200"))[1]
    assert [d["symbol"] for d in both.trade_decisions] == ["XOM", "APA"] and (both.trade_decisions[0]["min_price"], both.trade_decisions[0]["max_price"]) == (Decimal("160.0000"), Decimal("200.0000"))      # a range


def test_the_account_manager_tells_the_user_which_purchase_is_under_the_floor():
    said = []
    order = reviewed()
    order.data = wo.DataResult()
    manager(screen=said.append).decide_trades(order, BOB, None, MOMENT, "auto", "even", BUDGET, min_price=Decimal("160"))
    line = "EOG is at $158.32 a share, under your minimum of $160.00: it gets no decision and is not bought"
    assert any(line in said_line for said_line in said) and order.data.notes == [decisions.UNDER_FLOOR + line] and [d["symbol"] for d in order.trade_decisions] == ["XOM", "APA"]


def test_the_neutral_companies_are_priced_first_so_the_number_asked_for_is_filled_from_those_over_the_floor(monkeypatch):
    """The real neutral session `pam_smith_neutral.json`: MU ($778.31) and AMD ($534.06). With a floor of $600 AMD is left out, with a line for the user, and MU is the one decided; with one of $540 the same; with a range of $500 to $600, AMD."""
    monkeypatch.setenv("ALLOW_NEUTRAL_RECOMMENDATIONS", "true")
    said = []
    pam = UserProfile(user="Pam Smith", risk_tolerance="aggressive", age=30, trade_mode="auto")

    def run(**limits):
        am = AccountManager(model=object(), model_id="neutral", clerk=clerk(), profile=pam, fractional_even_split=False, screen=said.append)
        return am.decide_trades(neutral_only(), Portfolio(user="Pam Smith", cash_avail=Decimal("50000")), None, MOMENT, "auto", count=1, **limits)

    out = run(min_price=Decimal("600"))
    assert [(d["symbol"], d["will_trade"], d["origin"]) for d in out.trade_decisions] == [("MU", "yes_auto", "neutral")] and out.trade_decisions[0]["min_price"] == Decimal("600")
    assert any(line.strip().startswith("AMD is at $534.06 a share, under your minimum of $600.00") for line in said)
    assert [d["symbol"] for d in run(min_price=Decimal("500"), max_price=Decimal("600")).trade_decisions] == ["AMD"]      # MU ($778.31) is over the limit, AMD inside the range
    assert [d["symbol"] for d in run().trade_decisions] == ["MU"]


def trade(symbol="XOM", shares=8, side="buy"):
    return {"symbol": symbol, "side": side, "shares": shares, "shares_approved": shares}


def test_the_handshake_refuses_a_trade_whose_fill_price_is_under_the_floor_a_purchase_and_a_sale_and_never_one_at_or_over_it():
    held = BOB.model_copy(update={"cash_avail": Decimal("15000.00")})
    price = close_of("XOM")
    refused = handshake.decide_approval(held, trade(), price, MOMENT.date(), min_price=Decimal("200"))
    assert not refused.approved and refused.code == "not_permitted" and refused.shares == 0 and "under your floor of $200.00 a share" in refused.reason and "purchase is not made" in refused.reason
    assert handshake.decide_approval(held, trade(), price, MOMENT.date(), min_price=price).approved and handshake.decide_approval(held, trade(), price, MOMENT.date(), min_price=Decimal("100")).shares == 8
    assert handshake.decide_approval(held, trade(), price, MOMENT.date()).approved      # no floor
    jacob = jacob_as_bob()
    msft = close_of("MSFT")
    sale = handshake.decide_approval(jacob, trade("MSFT", 10, "sell"), msft, MOMENT.date(), min_price=msft + 1)
    assert not sale.approved and sale.code == "not_permitted" and "the sale is not made" in sale.reason
    assert handshake.decide_approval(jacob, trade("MSFT", 10, "sell"), msft, MOMENT.date(), min_price=msft).shares == 10 and handshake.decide_approval(jacob, trade("MSFT", 10, "sell"), msft, MOMENT.date(), max_price=Decimal("1")).shares == 10      # a sale is never held by a ceiling


def test_the_price_moved_below_the_floor_the_real_traders_purchase_is_rejected_at_the_handshake_and_nothing_is_bought_under_it():
    order = floored(Decimal("100"))[1]
    xom = next(d for d in order.trade_decisions if d["symbol"] == "XOM")
    xom["min_price"] = Decimal("200")      # as if the price had fallen under the user's floor since the decision
    am = manager()
    am.perform_trades(order, BOB, trader(), MOMENT)
    by_symbol = {t["symbol"]: t for t in order.trades}
    assert by_symbol["XOM"]["status"] == "rejected" and by_symbol["XOM"]["reject_code"] == "not_permitted" and "under your floor of $200.00 a share" in by_symbol["XOM"]["reason"]
    assert by_symbol["EOG"]["status"] == "filled" and wo.verify_trades(order) == []
    assert [p["symbol"] for p in order.trader_runs[-1]["portfolio_after"]["positions"]] == ["EOG"]


def test_a_fill_under_the_floor_is_an_output_defect_the_traders_own_check_names_and_a_floor_that_is_no_price_is_refused():
    order = floored(Decimal("100"))[1]
    manager().perform_trades(order, BOB, trader(), MOMENT)
    assert wo.verify_trades(order) == []
    filled = next(t for t in order.trades if t["symbol"] == "EOG" and t["status"] == "filled")
    next(d for d in order.trade_decisions if d["symbol"] == "EOG")["min_price"] = Decimal(filled["fill"]["price"]) + 1      # the floor was a dollar over what the fill cost
    assert any("under the user's price floor" in reason for reason in wo.verify_trades(order))
    bad = copy.deepcopy(order)
    next(d for d in bad.trade_decisions if d["symbol"] == "EOG")["min_price"] = "fifty"
    assert any("min_price must be a price above zero" in reason for reason in wo.verify_trade(bad, filled["record_id"], MOMENT))
    assert wo.decision_min_price({"min_price": None}) is None and wo.decision_min_price({}) is None and wo.decision_min_price({"min_price": 20}) == Decimal("20.0000")
    with pytest.raises(ValueError, match="min_price must be a price above zero"):
        wo.decision_min_price({"min_price": True})


def test_a_direct_order_record_may_carry_a_floor_and_it_is_checked_like_the_limit():
    good = {"record_id": "r1", "side": "sell", "symbol": "MSFT", "company": "Microsoft Corp", "shares": 10, "request": SELL_600, "requested_utc": "2026-11-13T20:30:00Z", "min_price": 600}
    assert wo.verify_direct_orders([good]) == [] and "min_price" in wo.DIRECT_OPTIONAL
    assert any("min_price is a price above zero" in reason for reason in wo.verify_direct_orders([{**good, "min_price": 0}]))


def test_the_floor_is_on_the_work_order_in_json_and_an_older_saved_order_has_none():
    order = wo.new_work_order("x", min_price=Decimal("20"))
    assert order.min_price == Decimal(20) and json.loads(order.model_dump_json())["min_price"] == 20 and wo.WorkOrder.model_validate_json(order.model_dump_json()).min_price == Decimal(20)
    older = json.loads(wo.new_work_order("x").model_dump_json())
    older.pop("min_price")
    assert wo.new_work_order("x").min_price is None and wo.WorkOrder.model_validate(older).min_price is None


def first_node(typed: str, key: str | None = None) -> dict:
    """The desk's first node over the real Account Manager (the replay of the model's real reply): the state when it has read the request."""
    request = {"ask": {"text": typed, "default_count": 3, "fetch": False}, "account": {"portfolio": BOB, "profile": AUTO, "mode": "auto"}}
    graph = desk.build_desk(clerk_factory=never("Data Clerk's node"), advisor_factory=never("Advisor"), critic_factory=never("Critic"), with_critic=True,
                            funds_factory=lambda: AccountManager(model=Replay(FORMS[key or typed]["replies"]), model_id="recorded", clerk=clerk()))
    state = None
    for state in graph.stream({"request": request, "order": None, "errors": [], "resolution": None, "funds": None, "path": [], "seconds": {}, "with_trader": False, "plan": None}, stream_mode="values"):
        if state["path"]:
            break
    return state


def test_the_desks_first_node_puts_the_floor_on_the_work_order_and_in_the_plan_and_the_typed_shares_in_the_plan():
    floor = first_node(OVER_20)
    assert floor["errors"] == [] and floor["order"].min_price == Decimal(20) and floor["order"].max_price is None and floor["plan"]["min_price"] == Decimal(20) and floor["plan"]["shares_of"] is None and floor["plan"]["shares_each"] is None
    shares = first_node(MU_AMD)
    assert shares["order"].min_price is None and shares["plan"]["shares_of"] == {"MU": Decimal(10), "AMD": Decimal(5)} and shares["plan"]["said_how_much"] is True
    each = first_node(EACH_5)
    assert each["plan"]["shares_each"] == Decimal(5) and each["plan"]["shares_of"] is None and each["plan"]["said_how_much"] is True and each["plan"]["count"] == 3 and each["plan"]["each"] is None


# ---------------------------------------------------------------- C1: a typed share count with a recommendation

def test_the_real_reply_to_buy_10_of_a_stock_if_the_recommendation_passes_is_a_purchase_of_that_stock_with_10_shares_as_its_limit():
    intent = reads(MU_PASSES_NEW)
    assert intent.action == "buy" and intent.symbols == ["MU"] and intent.shares_of == {"MU": Decimal(10)} and intent.shares_each is None and intent.budget is None and intent.corrected == [] and intent.said_how_much() is True
    assert intent.describe() == "buy the most recommended MU; if approved, 10 shares of MU" and type(intent.shares_of["MU"]) is Decimal
    older = read(MU_PASSES)      # the reply of the earlier prompt (action "order", the 10 on an order): the words still give the same reading
    assert older.action == "buy" and older.symbols == ["MU"] and older.shares_of == {"MU": Decimal(10)} and older.orders == [] and older.corrected == []
    assert json.loads(dumps(intent.model_dump()))["shares_of"] == {"MU": 10} and it.TradeIntent.model_validate(intent.model_dump()).shares_of == {"MU": Decimal(10)}


def test_the_real_reply_to_two_stocks_with_their_own_shares_reads_each_from_the_words():
    intent = reads(MU_AMD)
    assert intent.action == "buy" and intent.symbols == ["MU", "AMD"] and intent.shares_of == {"MU": Decimal(10), "AMD": Decimal(5)} and intent.said_how_much() is True
    assert intent.describe() == "buy the most recommended MU, AMD; if approved, 10 shares of MU and 5 shares of AMD"
    partial = it.parse_intent(form(action="buy", symbols=["MU", "AMD"]), "Buy 10 of MU, and AMD if they are recommended")
    assert partial.shares_of == {"MU": Decimal(10)} and partial.said_how_much() is False      # AMD has no number: Ask mode still asks how much


def test_the_real_reply_to_shares_for_each_of_the_top_3_is_one_number_for_every_purchase_and_the_each_is_not_money():
    intent = reads(EACH_5)
    assert (intent.action, intent.topic, intent.count, intent.shares_each, intent.shares_of, intent.budget, intent.each) == ("buy", "oil", 3, Decimal(5), {}, None, False) and intent.said_how_much() is True
    assert intent.describe() == "buy the 3 most recommended 'oil' companies; if approved, 5 shares of each" and "each" not in {k for k, v in intent.model_dump().items() if v is True}
    # the model read the 5 as the money for each, and `each` true with it: the words say shares, and code corrects it
    wrong = it.parse_intent(form(action="buy", topic="oil", count=3, budget=5, each=True), EACH_5)
    assert wrong.shares_each == Decimal(5) and wrong.budget is None and wrong.each is False and any("5 is a number of shares" in c for c in wrong.corrected)
    money = it.parse_intent(form(action="buy", topic="oil", count=3, budget=3000, each=True), "buy the top 3 oil stocks with $3,000 each if recommended")
    assert money.budget == Decimal(3000) and money.each is True and money.shares_each is None      # an amount of money for each is still money


@pytest.mark.parametrize("typed, expected", [
    ("buy 5 shares each of the top 3 oil stocks if recommended", (None, Decimal(5))), ("buy the top 3 oil stocks, 5 apiece, if recommended", (None, Decimal(5))), ("buy 2 of each of the best 3 oil stocks if recommended", (None, Decimal(2))),
    ("buy the top 3 oil stocks with $5 each if recommended", (None, None)), ("buy the top 3 oil stocks if recommended", (None, None)),
])
def test_the_number_for_every_stock_is_a_number_of_shares_and_nothing_written_as_money(typed, expected):
    topic = it.parse_intent(form(action="buy", topic="oil", count=3), typed)
    assert (topic.shares_of or None, topic.shares_each) == expected


def test_the_count_of_a_request_is_never_the_shares_and_a_fraction_or_two_numbers_are_for_the_user_to_clear_up():
    counted = it.parse_intent(form(action="buy", symbols=["MU"]), "Buy the 3 best MU if recommended")
    assert counted.shares_of == {} and counted.shares_each is None      # "3 best" is a count, not 3 shares
    with pytest.raises(it.NeedsClarification, match="whole shares"):
        it.parse_intent(form(action="buy", symbols=["MU"]), "Buy 0.5 shares of MU if the recommendation passes")
    with pytest.raises(it.NeedsClarification, match="more than one number of shares for MU"):
        it.parse_intent(form(action="buy", symbols=["MU"]), "Buy 10 shares of MU or 20 shares of MU if the recommendation passes")
    with pytest.raises(it.NeedsClarification, match="more than one number of shares for every stock"):
        it.parse_intent(form(action="buy", topic="oil", count=3), "buy 5 shares each of the top 3 oil stocks, or 6 apiece, if recommended")
    fault = it.parse_intent(form(action="buy", symbols=["MU"]), "Buy 10 shares of MU")      # a model that read a plain order as a purchase to be recommended: the 10 shares the user typed are still the limit (they were lost, and the cash was the only one)
    assert fault.action == "buy" and fault.shares_of == {"MU": Decimal(10)}


def test_a_number_of_shares_the_words_do_not_have_is_not_used_and_the_user_is_told():
    out = it.parse_intent(form(action="order", orders=[order_of("MU", 10)]), "Buy MU if the recommendation passes")
    assert out.action == "buy" and out.shares_of == {} and out.corrected == ["10 shares of MU is not a number of shares you typed, so it is not used"] and out.said_how_much() is False


def test_the_number_for_each_stock_and_a_number_for_one_that_differs_both_stand():
    intent = it.parse_intent(form(action="buy", symbols=["MU"], topic=None), "buy 5 shares of each, and 8 of MU if recommended")
    assert intent.shares_each == Decimal(5) and intent.shares_of == {"MU": Decimal(8)}


def test_a_stock_named_by_its_company_has_its_shares_too_and_a_ticker_matches_no_other_symbol():
    intent = it.parse_intent(form(action="buy", companies=["Micron"]), "Buy 10 of Micron if the recommendation passes")
    assert intent.shares_of == {"Micron": Decimal(10)} and intent.said_how_much() is True
    rows = [(decisions.sizing.Purchase(record_id="a", symbol="MU", action="open", price=Decimal("90"), rank=0), "Micron Technology Inc"),
            (decisions.sizing.Purchase(record_id="b", symbol="AMD", action="open", price=Decimal("90"), rank=1), "Advanced Micro Devices Inc")]
    assert decisions.typed_wanted(rows, {"Micron": Decimal(10)}) == {"a": Decimal(10)}      # the company's name finds its symbol
    assert decisions.typed_wanted(rows, {"MU": Decimal(10)}) == {"a": Decimal(10)} and decisions.typed_wanted(rows, {"AMD": Decimal(5)}) == {"b": Decimal(5)} and decisions.typed_wanted(rows, {"ZZZ": Decimal(5)}) == {}
    assert decisions.typed_wanted(rows, {"MU": Decimal(2)}, Decimal(5)) == {"a": Decimal(2), "b": Decimal(5)} and decisions.typed_wanted(rows, None, None) == {}
    assert decisions.combine_wanted({"a": Decimal(4)}, {"a": Decimal(3), "b": Decimal(1)}) == {"a": Decimal(3), "b": Decimal(1)} and decisions.combine_wanted(None, None) is None


# ---------------------------------------------------------------- C2: the typed shares are the approved purchase's limit, the funds the only check, a denied one buys nothing

def decide(shares_of=None, shares_each=None, cash=Decimal("15000.00"), mode="auto", **options):
    order = reviewed()
    am = manager(screen=lambda *lines: None)
    return am.decide_trades(order, BOB.model_copy(update={"cash_avail": cash}), options.pop("approver", None), MOMENT, mode, None, None, shares_of=shares_of, shares_each=shares_each, **options)


def test_the_typed_shares_of_an_approved_stock_are_its_limit_and_the_other_stock_is_sized_as_ever_and_the_denied_one_buys_nothing():
    plain = {d["symbol"]: d for d in decide().trade_decisions}
    assert plain["XOM"]["shares"] > 3 and plain["EOG"]["shares"] > 3
    out = {d["symbol"]: d for d in decide({"XOM": Decimal(3)}).trade_decisions}
    assert out["XOM"]["shares"] == 3 and out["XOM"]["size"]["binding"] == "asked" and out["XOM"]["size"]["asked"] == 3 and out["XOM"]["will_trade"] == "yes_auto"
    assert out["EOG"]["shares"] > 3 and "asked" not in out["EOG"]["size"]      # not typed: round robin over the cash
    assert out["APA"]["will_trade"] == "no_system_denied" and out["APA"]["shares"] == 0      # a Critic denial is final
    each = {d["symbol"]: d for d in decide(shares_each=Decimal(5)).trade_decisions}
    assert (each["XOM"]["shares"], each["EOG"]["shares"]) == (5, 5) and each["XOM"]["size"]["binding"] == "asked" == each["EOG"]["size"]["binding"] and each["APA"]["shares"] == 0
    mixed = {d["symbol"]: d for d in decide({"XOM": Decimal(2)}, Decimal(5)).trade_decisions}
    assert (mixed["XOM"]["shares"], mixed["EOG"]["shares"]) == (2, 5)      # a stock's own number first, the number for every stock for the rest


def test_with_too_little_cash_the_typed_shares_give_fewer_and_the_decision_says_what_held_them():
    out = {d["symbol"]: d for d in decide(shares_each=Decimal(5), cash=Decimal("500.00")).trade_decisions}      # XOM $170.59 and EOG $158.32: round robin, a share of each in turn
    assert (out["XOM"]["shares"], out["EOG"]["shares"]) == (2, 1) and out["XOM"]["size"]["binding"] == "money" and out["XOM"]["size"]["asked"] == 5
    assert sum(d["est_cost"] for d in out.values()) <= Decimal("500.00") and out["XOM"]["size"]["pot"] == Decimal("500.00")
    none = {d["symbol"]: d for d in decide({"XOM": Decimal(3)}, cash=Decimal("100.00")).trade_decisions}
    assert none["XOM"]["will_trade"] == "no_funds" and none["XOM"]["shares"] == 0      # not even one share: the funds are the only check


def test_the_typed_shares_are_told_to_the_user_and_a_stock_the_critic_did_not_approve_is_not_named():
    said = []
    order = reviewed()
    manager(screen=lambda *lines: said.extend(lines)).decide_trades(order, BOB, None, MOMENT, "auto", None, None, shares_of={"XOM": Decimal(3), "APA": Decimal(7)})
    assert any("How much, as you typed it: 3 shares of XOM (if approved; the cash is the only other limit)." in line for line in said) and not any("APA" in line and "typed" in line for line in said)
    assert {d["symbol"]: d["shares"] for d in order.trade_decisions}["APA"] == 0


def test_the_typed_shares_and_an_amount_of_money_for_each_are_both_the_users_and_the_smaller_wins():
    order = reviewed()
    out = manager(screen=lambda *lines: None).decide_trades(order, BOB, None, MOMENT, "auto", None, None, each=Decimal("1000"), shares_each=Decimal(2))
    by_symbol = {d["symbol"]: d for d in out.trade_decisions}
    assert (by_symbol["XOM"]["shares"], by_symbol["EOG"]["shares"]) == (2, 2)      # $1,000 buys 5 of each; 2 shares were typed
    assert by_symbol["XOM"]["size"]["pot"] == Decimal("2000.00")


def test_in_ask_mode_a_request_that_typed_the_shares_of_every_stock_asks_nothing_about_how_much_and_still_asks_to_approve_each_trade():
    intent = reads(MU_AMD)
    cards = []

    def approver(card):
        cards.append(card)
        return True

    channel = channel_of([])      # nobody is asked how much: it would fail the test by answering None
    order = reviewed()
    out = manager(screen=lambda *lines: None).decide_trades(order, BOB, approver, MOMENT, "ask", None, None, clarify=channel, said_how_much=intent.said_how_much(), shares_of={"XOM": Decimal(4), "EOG": Decimal(6)})
    assert channel.cards == [] and [c["symbol"] for c in cards] == ["XOM", "EOG"] and [c["shares"] for c in cards] == [4, 6]      # the question is the approval of each trade, with the typed shares as the final size (entry 79)
    assert {d["symbol"]: (d["will_trade"], d["shares"]) for d in out.trade_decisions}["XOM"] == ("yes_user", 4) and out.status == "decided"


def test_the_typed_shares_flow_through_the_real_trader_and_the_account_holds_exactly_them():
    order = reviewed()
    am = manager(screen=lambda *lines: None)
    am.decide_trades(order, BOB, None, MOMENT, "auto", None, None, shares_of={"XOM": Decimal(3)}, shares_each=Decimal(2))
    am.perform_trades(order, BOB, trader(), MOMENT)
    held = {p["symbol"]: p["shares"] for p in order.trader_runs[-1]["portfolio_after"]["positions"]}
    assert held == {"XOM": 3, "EOG": 2} and wo.verify_trades(order) == []
    assert [t["fill"]["shares"] for t in order.trades if t["status"] == "filled"] == [3, 2]


# ---------------------------------------------------------------- F3: a direct purchase with a floor is placed while the quote is at or over it (the waiting sale is part B's)

def test_the_real_desk_places_a_direct_purchase_whose_quote_is_over_its_floor_and_the_floor_is_on_the_decision_and_the_record():
    from test_request_numbers import run_desk
    state = run_desk(MSFT_OVER_20, AUTO)
    order = state["order"]
    assert state["errors"] == [] and state["path"] == ["account_manager", "account_manager_direct", "trader"] and order.status == "traded" and order.errors == []
    (decision,), (trade,) = order.trade_decisions, order.trades
    price = close_of("MSFT")
    assert price > 20 and decision["will_trade"] == "yes_user" and decision["origin"] == "direct" and decision["min_price"] == Decimal("20.0000") and order.direct_orders[0]["min_price"] == Decimal("20.0000")
    assert trade["status"] == "filled" and trade["fill"]["price"] >= 20 and wo.verify_trades(order) == [] and wo.verify_direct_orders(order.direct_orders) == []


# ---------------------------------------------------------------- the Time Machine reads the request as typed: the walk hands the floor on (the same code as a live run)

def test_the_walk_hands_the_floor_to_the_account_managers_decision_and_nothing_is_bought_under_it(tmp_path, no_model):
    """The REAL research of nine weekly decision days of a real fast Time Machine run (the oil request: FREE, no model, a day that is not in the cache fails the test): the same research walked with no floor and with a floor of $160 a share.
    The plan of each cached day carries it, as a live run's does; typed shares in the plan (`shares_each`) are read the same way and change no day's purchases of that run (they limit how many of each, and a hypothetical outcome is one share)."""
    free_run = walk(put_research(tmp_path / "free"))
    floored = walk(put_research(tmp_path / "floored", mutate=lambda entry: entry["plan"].__setitem__("min_price", 160)))
    each = walk(put_research(tmp_path / "each", mutate=lambda entry: entry["plan"].__setitem__("shares_each", 2)))

    def traded(run):
        return {(o.decision_day, o.symbol): o for o in run.outcomes if o.traded}

    a, b, c = traded(free_run), traded(floored), traded(each)
    assert free_run.status == floored.status == each.status == "finished" and a and b and c
    assert any(o.entry.price < 160 for o in a.values()) and all(o.entry.price >= Decimal(160) for o in b.values())      # with no floor it bought companies under $160 a share; with the floor, none (a neutral company over it may take the place of one under it)
    assert set(c) == set(a)      # the typed shares limit how many of each, not which
