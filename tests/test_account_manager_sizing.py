"""The Account Manager's sizing (`prometheus/agents/account_manager/sizing.py`, `configs/account_manager_v1.json`): how the user's own money is divided over the purchases, and the one check that stays: THE FUNDS COVER THE ORDER
(design choices entry 91; Jacob, 2026-11-15: "the only sizing that we need is to make sure the trade order has enough funds to cover it"). There is no limit of the profile, no loss cap, no order cap, no exposure room and no
liquidity cap any more, and nothing in this file tests one. Two allocations are tested here: the round robin, which is the default in BOTH modes
(a share of each purchase in turn, in rank order; design choices entry 106: there used to be a third way, "as much as the money allows, the first purchase first", that Ask mode chose when nothing was asked) and the even split
(equal dollars), with the budget the user gave and the amount the user asked for (`wanted`).

REAL DATA ONLY. The prices are the real closes of real symbols on 2026-11-13 (PG, MSFT, TSLA, GM, XOM, EOG), read from the real stored daily bars. The holdings are those of the real sample portfolio
`configs/portfolios/jacob_pyrett.json`. The $15,000 account is an amount of money, not stock data. No agent and no model is replaced by a stand-in: sizing has none. The tests are skipped when the stored bars are not on the machine.

Arithmetic is checked by hand: each expected number is written in the test from the real price and the money, not taken from the code under test.
"""
from decimal import Decimal
import ast
import json
import math
import os
from datetime import date, timedelta
from pathlib import Path

import pytest

from prometheus.agents.account_manager import config as amc
from prometheus.agents.account_manager import sizing as sz
from prometheus.common.money import cents, price as door_price
from prometheus.common.portfolio import load_portfolio, trade_amount
from prometheus.rag.config import REPO_ROOT
from prometheus.tools.data_clerk import DataClerkTools

DAY = date(2026, 11, 13)
SYMBOLS = ["PG", "MSFT", "TSLA", "GM", "XOM", "EOG"]
CASH = Decimal("15000")
SAMPLE = REPO_ROOT / "configs" / "portfolios" / "jacob_pyrett.json"
CONFIG = amc.load_config()


@pytest.fixture(scope="module")
def market():
    """{symbol: {price}} from the real stored bars of 2026-11-13: the close."""
    try:
        found = DataClerkTools().daily_bars(SYMBOLS, DAY - timedelta(days=30), DAY)["bars"]
    except FileNotFoundError:
        pytest.skip("the stored daily bars are not on this machine")
    out = {}
    for symbol in SYMBOLS:
        rows = found.get(symbol) or []
        if not rows or rows[-1]["date"] != DAY.isoformat():
            pytest.skip(f"the stored bars do not hold {DAY} for {symbol}")
        out[symbol] = {"price": door_price(rows[-1]["close"])}                 # a stored bar is a float: it comes into the account's code as a Decimal through the price door, once
    return out


def buy(symbol, market, rank=0, **changes) -> sz.Purchase:
    return sz.Purchase(**{"record_id": f"rec-{symbol}", "symbol": symbol, "action": "open", "price": market[symbol]["price"], "rank": rank, **changes})


def size(purchases, cash=CASH, config=CONFIG, **options) -> dict[str, sz.Sized]:
    """{record_id: Sized}"""
    return {s.record_id: s for s in sz.size_purchases(purchases, cash, config, **options)}


# ---------------------------------------------------------------- the money, and nothing else, bounds a purchase


def test_an_even_split_is_held_down_by_a_smaller_amount_the_user_asked_for_on_one_purchase(market):
    pg, gm = market["PG"]["price"], market["GM"]["price"]
    got = size([buy("PG", market, rank=1), buy("GM", market, rank=2)], cash=Decimal("1000"), allocation="even", wanted={"rec-PG": 1})
    assert (got["rec-PG"].shares, got["rec-PG"].binding) == (1, "asked")
    assert got["rec-GM"].binding == "even_split" and got["rec-GM"].shares == math.floor(pg / gm) and got["rec-GM"].target_dollars == cents(pg)
    assert "not spent" in got["rec-GM"].notes[-1] and all("more share" not in n for n in got["rec-GM"].notes)       # entry 108: a smaller ask set the equal amount, so the rest of the $1,000 is not the split's to spend (step 2 does not run)


# ---------------------------------------------------------------- the round robin (Auto mode: Jacob, 2026-11-14)
# "Buy a share of the first stock, buy a share of the next stock, and keep buying shares until all the desired funds are used up." Each stock ends with a FINAL number of shares.


def test_a_purchase_that_cannot_be_sized_is_left_out_of_the_round_robin_and_does_not_stop_the_others(market):
    got = size([buy("GM", market, rank=1, price=None), buy("XOM", market, rank=2), buy("EOG", market, rank=3)], allocation="round_robin", budget=Decimal("500"))
    assert (got["rec-GM"].shares, got["rec-GM"].binding) == (0, "no_room") and "not sized: no price" in got["rec-GM"].notes[-1]
    assert (got["rec-XOM"].shares, got["rec-EOG"].shares) == (2, 1)


# ---------------------------------------------------------------- what only a program can get wrong is an error

def test_arguments_that_make_no_sense_are_refused(market):
    one = [buy("GM", market)]
    for options in ({"allocation": "random"}, {"budget": 0}, {"budget": Decimal("-5")}, {"budget": True}, {"budget": Decimal("NaN")}, {"budget": Decimal("Infinity")}, {"budget": "100"}, {"cash": Decimal("-1")}, {"cash": Decimal("NaN")},
                    {"wanted": {"rec-NONE": 1}}, {"wanted": {"rec-GM": -1}}, {"wanted": {"rec-GM": 0.1234567}}, {"wanted": {"rec-GM": True}}, {"wanted": {"rec-GM": "2"}}):      # AMENDED (entry 111, M3): 1.5 is a quantity now (what the user's dollars for each stock of a direct order buy); seven decimals and a string are still refused
        with pytest.raises(ValueError):
            size(one, **options)
    with pytest.raises(ValueError, match="same record_id"):
        size([buy("GM", market), buy("GM", market)])


# ---------------------------------------------------------------- the funds check

def test_the_cash_covers_a_small_purchase(market):
    portfolio = load_portfolio(SAMPLE)
    ok, note = sz.funds_check(Decimal("40"), portfolio, {s: market[s]["price"] for s in ("MSFT", "TSLA", "GM")}, DAY)
    assert ok is True and note == "cash $51.23 covers $40.00"


# ---------------------------------------------------------------- the settings file


def test_the_settings_file_must_exist_and_be_json(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        amc.load_config(tmp_path / "missing.json")
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        amc.load_config(tmp_path / "broken.json")


def test_every_threshold_of_the_sizing_is_in_the_settings_file_not_in_the_code():
    source = (REPO_ROOT / "prometheus" / "agents" / "account_manager" / "sizing.py").read_text(encoding="utf-8")
    numbers = {node.value for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool)}
    assert numbers <= {0, 1}, numbers                                                          # a 0 or a 1 is a count or a bound; there is no 0.05, 0.1, 1e-9 or other threshold in the module


# ---------------------------------------------------------------- the module's own imports, and what it no longer knows

def imports_of(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            found |= {node.module or "", *(f"{node.module}.{a.name}" for a in node.names)}
    return found


def test_the_sizing_imports_nothing_of_the_critic_the_advisor_a_model_or_the_network():
    path = REPO_ROOT / "prometheus" / "agents" / "account_manager" / "sizing.py"
    names = imports_of(path)
    forbidden = ("prometheus.agents.critic", "prometheus.agents.advisor", "prometheus.agents.data_clerk", "prometheus.agents.trader", "prometheus.tools.data_clerk", "prometheus.models", "prometheus.langchain",
                 "prometheus.langgraph", "prometheus.prompts", "prometheus.rag", "prometheus.reasoning", "prometheus.persistence", "prometheus.agents.account_manager.decisions", "prometheus.agents.account_manager.account_manager",
                 "prometheus.common.user_profile", "langchain", "langchain_core", "langgraph", "openai", "anthropic", "httpx", "requests", "urllib", "socket", "aiohttp", "psycopg", "psycopg2", "subprocess", "os", "dotenv")
    for name in names:
        assert not any(name == bad or name.startswith(bad + ".") for bad in forbidden), f"sizing.py imports {name}"
    own = {n for n in names if n.startswith("prometheus")}
    # AMENDED (entry 110): the fractional even split reads `shares.buys` and `shares.Shares` (plain Python, no model, no network), so `prometheus.common.shares` is one of the module's own imports
    assert own <= {"prometheus.agents.account_manager.config", "prometheus.agents.account_manager.config.AccountManagerConfig", "prometheus.agents.account_manager.config.load_config",
                   "prometheus.common", "prometheus.common.shares", "prometheus.common.money", "prometheus.common.money.CENT", "prometheus.common.money.NonNegativeMoney", "prometheus.common.money.Money",
                   "prometheus.common.money.Price", "prometheus.common.money.exact", "prometheus.common.portfolio", "prometheus.common.portfolio.Portfolio", "prometheus.common.portfolio.trade_amount", "prometheus.tools", "prometheus.tools.trading_math",
                   "prometheus.common.trace", "prometheus.common.trace.traced"}, own      # AMENDED (entry 115): the span decorator `traced` (plain Python over `langsmith`: no model, no network of its own, and it makes no run unless a trace is open)


def test_the_sizing_knows_no_limit_of_the_profile_and_no_stop():
    """Entry 91: the only check is that the funds cover the order. The module, the profile and the settings file name none of the things that used to cap a purchase."""
    gone = {"max_position_pct", "max_loss_per_trade_pct", "max_new_exposure_pct", "max_order_dollars", "SizingLimits", "sizing_limits", "limit_overrides", "stop_distance", "held_shares", "LIMIT_CAPS"}
    for path in (REPO_ROOT / "prometheus" / "agents" / "account_manager" / "sizing.py", REPO_ROOT / "prometheus" / "common" / "user_profile.py", REPO_ROOT / "prometheus" / "agents" / "account_manager" / "config.py",
                 REPO_ROOT / "prometheus" / "agents" / "account_manager" / "decisions.py", REPO_ROOT / "prometheus" / "agents" / "account_manager" / "clarify.py"):
        used = {n.id for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))) if isinstance(n, ast.Attribute)}
        assert used & gone == set(), (path.name, used & gone)
    from prometheus.common.user_profile import UserProfile
    assert not gone & set(UserProfile.model_fields)
    with pytest.raises(ValueError, match="max_position_pct"):                                   # a profile that still carries one is refused, not silently ignored
        UserProfile.model_validate({"user": "A B", "risk_tolerance": "moderate", "max_position_pct": 0.1})


def test_the_settings_loader_reads_nothing_but_the_file():
    names = imports_of(REPO_ROOT / "prometheus" / "agents" / "account_manager" / "config.py")
    assert {n for n in names if n.startswith("prometheus")} <= {"prometheus.rag.config", "prometheus.rag.config.REPO_ROOT"}
    assert not any(n.split(".")[0] in ("langchain", "langchain_core", "openai", "httpx", "requests") for n in names)
