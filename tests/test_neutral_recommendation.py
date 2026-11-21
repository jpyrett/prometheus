"""A NEUTRAL RECOMMENDATION IS APPROVED (design choices entry 91; Jacob, 2026-11-15, with `ALLOW_NEUTRAL_RECOMMENDATIONS=true` in `.env`), and no profile limit applies to anything.

REAL DATA ONLY. `tests/fixtures/session_real/pam_smith_neutral.json` is a capture of the real session `./test_prometheus.sh --session --account <Pam Smith> --yes --prompt "Buy MU, SNDK and AMD if the recommendations pass. Go ahead
and trade if the recommendations pass."` (2026-11-15; Pam Smith, 30, aggressive, $50,000, Auto mode; the paid refresh of the data; the real Account Manager, Data Clerk, Advisor, Critic (GLM 5.3) and Trader (gpt-oss-120b), the
real Postgres book, the simulated clock of Friday 2026-11-13): SNDK was bullish and the Critic DENIED it (risk above the profile, a falling knife: final), MU and AMD were scored NEUTRAL by the Advisor (no strategy passed), so
they were treated as approved and bought in equal dollars (entry 91's default for a run of neutral companies alone): MU 32 shares, AMD 46, $502.59 left. Since entry 109 (Jacob, 2026-11-15: *"use round-robin for neutral-only runs too. Use split evenly
if the user specifies it in the prompt"*) the same request, which did not ask for an even split, is spread round-robin (MU 38, AMD 38) and equal dollars come only with the user's words ("split evenly"). The tests re-decide that order with the real Account Manager over the real stored closes (the Data Clerk's quotes) and trade it with
the real Trader, and break the permission one field at a time. No agent and no model is replaced by a stand-in. The tests are skipped when the stored bars are not on the machine.
"""
from decimal import Decimal
import copy
import inspect
import json
from pathlib import Path

import pytest
from handshake_helpers import FRIDAY, MOMENT, close_of, friday_close, trader  # noqa: F401  (friday_close is a fixture)
from test_account_manager_clarify import RECORDED, Replay, channel_of

from prometheus.agents.account_manager import neutral
from prometheus.agents.account_manager.account_manager import AccountManager
from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
from prometheus.common import shares as sh
from prometheus.common import work_order as wo
from prometheus.common.portfolio import Portfolio
from prometheus.common.user_profile import UserProfile

pytestmark = pytest.mark.usefixtures("friday_close")

CAPTURE = json.loads((Path(__file__).parent / "fixtures" / "session_real" / "pam_smith_neutral.json").read_text(encoding="utf-8"))
PAM = Portfolio(user="Pam Smith", cash_avail=Decimal("50000"))
PAM_PROFILE = UserProfile(user="Pam Smith", risk_tolerance="aggressive", age=30, trade_mode="auto")


@pytest.fixture
def allowed(monkeypatch):
    monkeypatch.setenv("ALLOW_NEUTRAL_RECOMMENDATIONS", "true")


def advised() -> wo.WorkOrder:
    """The real work order as it was when the Critic had finished and before the Account Manager decided: the Advisor's recommendation (SNDK) and its candidates (MU and AMD, neutral), the Critic's denial of SNDK."""
    order = wo.WorkOrder.model_validate(copy.deepcopy(CAPTURE["order"]))
    order.status, order.trade_decisions, order.trades, order.trader_runs = "reviewed", [], [], []
    return order


def neutral_only() -> wo.WorkOrder:
    """The same order with nothing for the Critic to review: the Advisor recommended no company, and two were neutral."""
    order = advised()
    order.recommendations, order.reviews, order.status = [], [], "recommended"
    return order


def account(**options) -> AccountManager:
    # AMENDED (entry 110): `fractional_even_split` is False here, so that every test of this file keeps pinning the WHOLE-share even split of entry 108 (the Account Manager splits in fractions unless its switch is off);
    # the fractional split of the same real oil session is in tests/test_fractional_even_split.py
    options.setdefault("fractional_even_split", False)
    return AccountManager(model=options.pop("model", object()), model_id="neutral", clerk=DataClerk(model=object()), profile=options.pop("profile", PAM_PROFILE), **options)


def decided(order, mode="auto", approver=None, portfolio=PAM, **kw):
    return account().decide_trades(order, portfolio, approver, MOMENT, mode, **kw)


# ---------------------------------------------------------------- Ask mode: the user is asked for the number before anything is proposed (Jacob, 2026-11-15)

REQUEST = "I want to buy MU and AMD. Go ahead and trade if the recommendations pass."          # no amount in it: "I want to buy <X>"


def ask_how_much(scenario: str, said_how_much: bool = False, mode: str = "ask"):
    """Pam's neutral companies in a mode, with the model's REAL recorded replies to a user who answers `scenario`'s words (tests/fixtures/session_real/clarify_ask_mode.json)."""
    recorded = RECORDED[scenario]
    model, channel, cards = Replay(recorded["replies"]), channel_of(recorded["answers"]), []
    out = account(model=model).decide_trades(neutral_only(), PAM, lambda card: cards.append(card) or True, MOMENT, mode, clarify=channel, said_how_much=said_how_much, request_text=REQUEST)
    return out, channel, cards, model


def test_a_number_of_shares_the_user_gives_is_the_most_of_each_neutral_company_that_is_bought(allowed):
    out, channel, cards, _ = ask_how_much("shares_each_raise_yes")                                                                 # "10 shares of each"
    assert [(d["symbol"], d["shares"], d["size"]["asked"]) for d in out.trade_decisions] == [("MU", 10, 10), ("AMD", 10, 10)]
    assert sum(d["est_cost"] for d in out.trade_decisions) < Decimal("50000") and [c["shares"] for c in cards] == [10, 10]


# ---------------------------------------------------------------- the Trader's permission


def test_the_word_neutral_alone_permits_nothing(allowed):
    base = decided(advised())
    mu = base.trade_decisions[1]
    gone = copy.deepcopy(base)
    gone.candidates = [c for c in gone.candidates if c["symbol"] != "MU"]
    assert "holds no neutral score of the Advisor's for 'MU'" in "; ".join(wo.verify_trade(gone, mu["record_id"], MOMENT))
    bullish = copy.deepcopy(base)
    next(c for c in bullish.candidates if c["symbol"] == "MU")["stance"] = "bullish"
    assert "holds no neutral score" in "; ".join(wo.verify_trade(bullish, mu["record_id"], MOMENT))
    gated = copy.deepcopy(base)
    next(c for c in gated.candidates if c["symbol"] == "MU")["gate"] = "bars_stale"
    assert "holds no neutral score" in "; ".join(wo.verify_trade(gated, mu["record_id"], MOMENT))
    sale = copy.deepcopy(base)
    sale.trade_decisions[1]["action"] = "reduce"
    assert "a neutral recommendation is only a purchase" in "; ".join(wo.verify_trade(sale, mu["record_id"], MOMENT))
    reviewed = copy.deepcopy(base)
    reviewed.reviews = [*reviewed.reviews, {"record_id": mu["record_id"], "approved": True}]
    assert "a neutral recommendation has no Critic review" in "; ".join(wo.verify_trade(reviewed, mu["record_id"], MOMENT))
    nobody = copy.deepcopy(base)
    nobody.trade_decisions[1]["will_trade"] = "pending_user"
    assert "not one of yes_auto, yes_user" in "; ".join(wo.verify_trade(nobody, mu["record_id"], MOMENT))
    del base.trade_decisions[1]["origin"]
    assert wo.verify_trade(base, mu["record_id"], MOMENT) == [f"there is no Critic review of recommendation {mu['record_id']}"]            # not neutral any more: the old rule


# ---------------------------------------------------------------- Part E: a neutral recommendation is an approval, in full (entry 107)

OIL_CAPTURE = json.loads((Path(__file__).parent / "fixtures" / "session_real" / "auto_oil_no_second_mark.json").read_text(encoding="utf-8"))
OIL_CASH = Portfolio(user="Auto Mode Check", cash_avail=Decimal("15000"))
OIL_USER = UserProfile(user="Auto Mode Check", risk_tolerance="conservative", age=60, trade_mode="auto")


def oil_order() -> wo.WorkOrder:
    """The real oil session (`auto_oil_no_second_mark.json`) as it was when the Critic had finished: EOG 79 and CVX 86 approved, COP denied, six companies the Advisor scored neutral."""
    order = wo.WorkOrder.model_validate(copy.deepcopy(OIL_CAPTURE["order"]))
    order.status, order.trade_decisions, order.trades, order.trader_runs = "reviewed", [], [], []
    return order
