"""The Account Manager's sizing (`prometheus/agents/account_manager/sizing.py`, `configs/account_manager_v1.json`): how the user's own money is divided over the purchases, and the one check that stays: THE FUNDS COVER THE ORDER
(design choices entry 91; Jacob, 2026-11-15: "the only sizing that we need is to make sure the trade order has enough funds to cover it"). There is no limit of the profile, no loss cap, no order cap, no exposure room and no
liquidity cap any more, and nothing in this file tests one. The three allocations are tested here: the default (as much as the money allows, in rank order), the even split (equal dollars) and the round robin (a share of
each purchase in turn, in rank order: Auto mode's way), with the budget the user gave and the amount the user asked for (`wanted`).

REAL DATA ONLY. The prices are the real closes of real symbols on 2026-11-13 (PG, MSFT, TSLA, GM, XOM, EOG), read from the real stored daily bars. The holdings are those of the real sample portfolio
`configs/portfolios/jacob_pyrett.json`. The $15,000 account is an amount of money, not stock data. No agent and no model is replaced by a stand-in: sizing has none. The tests are skipped when the stored bars are not on the machine.

Arithmetic is checked by hand: each expected number is written in the test from the real price and the money, not taken from the code under test.
"""
import ast
import json
import math
import os
from datetime import date, timedelta
from pathlib import Path

import pytest

from prometheus.agents.account_manager import config as amc
from prometheus.agents.account_manager import sizing as sz
from prometheus.common.portfolio import load_portfolio, trade_amount
from prometheus.rag.config import REPO_ROOT
from prometheus.tools.data_clerk import DataClerkTools

DAY = date(2026, 11, 13)
SYMBOLS = ["PG", "MSFT", "TSLA", "GM", "XOM", "EOG"]
CASH = 15000.0
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
        out[symbol] = {"price": rows[-1]["close"]}
    return out


def buy(symbol, market, rank=0, **changes) -> sz.Purchase:
    return sz.Purchase(**{"record_id": f"rec-{symbol}", "symbol": symbol, "action": "open", "price": market[symbol]["price"], "rank": rank, **changes})


def size(purchases, cash=CASH, config=CONFIG, **options) -> dict[str, sz.Sized]:
    """{record_id: Sized}"""
    return {s.record_id: s for s in sz.size_purchases(purchases, cash, config, **options)}


def settings_file(tmp_path, **sizing) -> Path:
    """The shipped settings file with some of its `sizing` numbers changed (a threshold, not stock data)."""
    data = json.loads(amc.CONFIG_PATH.read_text(encoding="utf-8"))
    data["sizing"].update(sizing)
    path = tmp_path / "account_manager.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------- the money, and nothing else, bounds a purchase

def test_a_purchase_gets_the_whole_shares_the_cash_pays_for_and_the_cash_is_not_reserved_for_the_next(market):
    pg, gm = market["PG"]["price"], market["GM"]["price"]
    got = size([buy("PG", market, rank=1), buy("GM", market, rank=2)])
    assert (got["rec-PG"].shares, got["rec-PG"].binding) == (math.floor(CASH / pg), "money") and got["rec-PG"].est_cost == trade_amount(got["rec-PG"].shares, pg)
    assert (got["rec-GM"].shares, got["rec-GM"].binding) == (math.floor(CASH / gm), "money")                          # each sees ALL the cash: the Trader's handshake meets the cash as it is at each trade
    assert set(got["rec-PG"].caps) == {"money", "budget"} and got["rec-PG"].caps["budget"] is None


def test_a_whole_number_of_shares_that_the_float_makes_a_hair_short_is_still_bought(market):
    price = market["PG"]["price"]
    money = round(3 * price, 2)                                                                 # exactly three shares of PG, in cents
    assert money / price < 3 and math.floor(money / price) == 2                                 # and the division is a hair short of 3: without the guard, two shares
    for allocation in (None, "even"):
        s = size([buy("PG", market)], cash=money, allocation=allocation)["rec-PG"]
        assert s.shares == 3 and s.binding == "money" and s.est_cost == money
    s = size([buy("PG", market)], budget=money)["rec-PG"]
    assert s.shares == 3 and s.binding == "budget"


def test_the_budget_the_user_gave_is_shared_in_rank_order_while_the_cash_is_not(market):
    pg, gm = market["PG"]["price"], market["GM"]["price"]
    got = size([buy("PG", market, rank=1), buy("GM", market, rank=2)], budget=1000.0)
    first = math.floor(1000 / pg)
    assert (got["rec-PG"].shares, got["rec-PG"].binding) == (first, "budget")
    left = 1000.0 - trade_amount(first, pg)
    assert (got["rec-GM"].shares, got["rec-GM"].binding) == (math.floor(left / gm), "budget") and got["rec-GM"].shares >= 1


def test_the_amount_the_user_asked_for_is_the_most_a_purchase_gets_and_is_said(market):
    pg = market["PG"]["price"]
    s = size([buy("PG", market)], wanted={"rec-PG": 3})["rec-PG"]
    assert (s.shares, s.binding, s.asked) == (3, "asked", 3) and s.est_cost == trade_amount(3, pg)
    more_than_the_cash = size([buy("PG", market)], wanted={"rec-PG": 10_000})["rec-PG"]
    assert (more_than_the_cash.shares, more_than_the_cash.binding) == (math.floor(CASH / pg), "money") and more_than_the_cash.asked == 10_000       # the cash is the only cap: asking for more does not buy more
    assert size([buy("PG", market)], wanted={"rec-PG": 0})["rec-PG"].shares == 0


def test_an_even_split_records_the_equal_amount_and_how_many_shared_it(market):
    ps = [buy("PG", market, rank=1), buy("GM", market, rank=2), buy("MSFT", market, rank=3)]
    even = size(ps, allocation="even")
    assert all(s.allocation == "even" and s.of == 3 and s.target_dollars == round(CASH / 3, 2) for s in even.values())
    assert all(s.shares == math.floor((CASH / 3) / market[i[4:]]["price"]) for i, s in even.items())
    assert all(s.allocation == "as_much_as_possible" and s.target_dollars is None and s.of is None for s in size(ps).values())
    dropped = size([buy("GM", market, rank=1), buy("PG", market, rank=2), buy("TSLA", market, rank=3)], cash=600.0, allocation="even")
    assert dropped["rec-TSLA"].shares == 0 and dropped["rec-TSLA"].target_dollars == 200.0 and dropped["rec-TSLA"].of == 3 and dropped["rec-GM"].target_dollars == 300.0 and dropped["rec-GM"].of == 2


def test_an_even_split_is_held_down_by_a_smaller_amount_the_user_asked_for_on_one_purchase(market):
    pg, gm = market["PG"]["price"], market["GM"]["price"]
    got = size([buy("PG", market, rank=1), buy("GM", market, rank=2)], cash=1000.0, allocation="even", wanted={"rec-PG": 1})
    assert (got["rec-PG"].shares, got["rec-PG"].binding) == (1, "asked")
    assert got["rec-GM"].binding == "even_split" and got["rec-GM"].shares == math.floor(pg / gm) and got["rec-GM"].target_dollars == round(pg, 2)


# ---------------------------------------------------------------- the round robin (Auto mode: Jacob, 2026-11-14)
# "Buy a share of the first stock, buy a share of the next stock, and keep buying shares until all the desired funds are used up." Each stock ends with a FINAL number of shares.

def test_a_stock_the_user_asked_a_few_shares_of_stops_alone_while_the_others_go_on(market):
    """PG is asked for 2 shares: rounds 1 and 2 take a share of each; from round 3 PG is closed ('asked') and GM goes on alone until the pot has no money for another share."""
    pg, gm = market["PG"]["price"], market["GM"]["price"]
    got = size([buy("PG", market, rank=1), buy("GM", market, rank=2)], allocation="round_robin", budget=1000.0, wanted={"rec-PG": 2})
    assert (got["rec-PG"].shares, got["rec-PG"].binding) == (2, "asked") and got["rec-PG"].est_cost == trade_amount(2, pg)
    assert got["rec-PG"].notes[1] == "closed at 2 shares: it has the shares the user asked for"
    assert (got["rec-GM"].shares, got["rec-GM"].binding) == (math.floor((1000.0 - trade_amount(2, pg)) / gm), "budget")


def test_a_purchase_the_pot_cannot_buy_one_share_of_has_no_room_and_the_others_still_go_on(market):
    """XOM is ranked first and a pot of $100 buys none of it; GM (86.00) is behind it and takes one share. Closed by 'budget': the note says so."""
    xom, gm = market["XOM"]["price"], market["GM"]["price"]
    assert xom > 100 > gm
    got = size([buy("XOM", market, rank=1), buy("GM", market, rank=2)], allocation="round_robin", budget=100.0)
    assert (got["rec-XOM"].shares, got["rec-XOM"].est_cost, got["rec-XOM"].binding) == (0, 0.0, "no_room") and "not even one share fits: the budget has no money for another share" in got["rec-XOM"].notes[-1]
    assert (got["rec-GM"].shares, got["rec-GM"].binding, got["rec-GM"].est_cost) == (1, "budget", gm)


def test_a_purchase_that_cannot_be_sized_is_left_out_of_the_round_robin_and_does_not_stop_the_others(market):
    got = size([buy("GM", market, rank=1, price=None), buy("XOM", market, rank=2), buy("EOG", market, rank=3)], allocation="round_robin", budget=500.0)
    assert (got["rec-GM"].shares, got["rec-GM"].binding) == (0, "no_room") and "not sized: no price" in got["rec-GM"].notes[-1]
    assert (got["rec-XOM"].shares, got["rec-EOG"].shares) == (2, 1)


def test_with_the_whole_cash_as_the_pot_every_stock_ends_with_shares_and_the_spending_is_exact(market):
    ps = [buy("XOM", market, rank=1), buy("EOG", market, rank=2), buy("PG", market, rank=3)]
    got = size(ps, allocation="round_robin")
    spent = sum(s.est_cost for s in got.values())
    assert all(s.shares > 1 and s.binding == "money" and s.est_cost == trade_amount(s.shares, market[i[4:]]["price"]) for i, s in got.items()) and spent <= CASH
    assert CASH - spent < min(market[s]["price"] for s in ("XOM", "EOG", "PG"))                      # what is left buys no share of any of them
    assert max(s.shares for s in got.values()) - min(s.shares for s in got.values()) <= 3 + math.ceil(CASH / 3 / market["PG"]["price"])     # a share of each in turn: the dearer stock has fewer shares only by price


def test_the_round_robin_never_spends_more_than_the_pot_and_stops_under_a_share_over_a_range_of_real_pots(market):
    ps = [buy("XOM", market, rank=1), buy("EOG", market, rank=2), buy("PG", market, rank=3), buy("GM", market, rank=4)]
    prices = {f"rec-{s}": market[s]["price"] for s in ("XOM", "EOG", "PG", "GM")}
    for pot in range(50, 4000, 97):
        got = size(ps, allocation="round_robin", budget=float(pot))
        spent = sum(s.est_cost for s in got.values())
        assert spent <= pot + 1e-9 and all(s.est_cost == trade_amount(s.shares, prices[i]) for i, s in got.items())
        closed_by_the_pot = [i for i, s in got.items() if s.binding in ("budget", "no_room")]
        assert closed_by_the_pot and pot - spent < min(prices[i] for i in closed_by_the_pot) + 1e-9                    # no share that was refused would have fitted what is left
        assert all(s.binding in sz.BINDINGS and isinstance(s.shares, int) for s in got.values())


def test_the_allocations_are_the_three_names_and_round_robin_is_one_of_them():
    assert sz.ALLOCATIONS == ("even", "round_robin", "as_much_as_possible")


# ---------------------------------------------------------------- what cannot be sized is said, and never guessed

def test_no_price_means_no_size(market):
    for price in (None, 0.0, -1.0, float("nan")):
        s = size([buy("GM", market, price=price)])["rec-GM"]
        assert (s.shares, s.est_cost, s.binding) == (0, 0.0, "no_room") and set(s.caps) == set(sz.CAP_NAMES) and s.caps["money"] is None
        assert "not sized: no price" in s.notes[-1]


def test_no_cash_means_no_share_and_says_so(market):
    s = size([buy("GM", market)], cash=0.0)["rec-GM"]
    assert (s.shares, s.binding) == (0, "no_room") and "not even one share fits" in s.notes[-1]


# ---------------------------------------------------------------- what only a program can get wrong is an error

def test_arguments_that_make_no_sense_are_refused(market):
    one = [buy("GM", market)]
    for options in ({"allocation": "random"}, {"budget": 0}, {"budget": -5.0}, {"budget": True}, {"budget": float("nan")}, {"budget": float("inf")}, {"budget": "100"}, {"cash": -1.0}, {"cash": float("nan")},
                    {"wanted": {"rec-NONE": 1}}, {"wanted": {"rec-GM": -1}}, {"wanted": {"rec-GM": 1.5}}, {"wanted": {"rec-GM": True}}):
        with pytest.raises(ValueError):
            size(one, **options)
    with pytest.raises(ValueError, match="same record_id"):
        size([buy("GM", market), buy("GM", market)])


# ---------------------------------------------------------------- the funds check

def test_the_cash_covers_a_small_purchase(market):
    portfolio = load_portfolio(SAMPLE)
    ok, note = sz.funds_check(40.0, portfolio, {s: market[s]["price"] for s in ("MSFT", "TSLA", "GM")}, DAY)
    assert ok is True and note == "cash $51.23 covers $40.00"


def test_holdings_in_profit_could_raise_what_the_cash_does_not_cover(market):
    portfolio = load_portfolio(SAMPLE)
    prices = {s: market[s]["price"] for s in ("MSFT", "TSLA", "GM")}
    raisable = 49 * prices["MSFT"] + 122 * prices["TSLA"] + 59 * prices["GM"]                  # every holding is in profit and old enough to sell, and one share of each is kept
    assert all(p.cost_basis < prices[p.symbol] for p in portfolio.positions)
    ok, note = sz.funds_check(5000.0, portfolio, prices, DAY)
    assert ok is True and note == f"short by ${5000.0 - 51.23:,.2f}; ${raisable:,.2f} could be raised from holdings in profit that are sellable today (one share of each kept)"
    ok, note = sz.funds_check(raisable + 51.23 + 1.0, portfolio, prices, DAY)                   # a dollar more than everything that could be raised
    assert ok is False and note.startswith("short by $")


def test_a_holding_in_the_red_or_locked_by_t_plus_one_is_not_counted(market):
    portfolio = load_portfolio(SAMPLE)
    prices = {s: market[s]["price"] for s in ("MSFT", "TSLA", "GM")}
    red = portfolio.model_copy(update={"positions": [p.model_copy(update={"cost_basis": prices["TSLA"] * 2}) if p.symbol == "TSLA" else p for p in portfolio.positions]})    # TSLA bought dear: in the red
    _, note = sz.funds_check(5000.0, red, prices, DAY)
    assert f"${49 * prices['MSFT'] + 59 * prices['GM']:,.2f} could be raised" in note
    locked = portfolio.model_copy(update={"positions": [p.model_copy(update={"acquired": DAY}) if p.symbol == "GM" else p for p in portfolio.positions]})              # GM bought today: not sellable
    _, note = sz.funds_check(5000.0, locked, prices, DAY)
    assert f"${49 * prices['MSFT'] + 122 * prices['TSLA']:,.2f} could be raised" in note


def test_a_holding_with_no_price_is_not_counted_and_the_note_names_it(market):
    portfolio = load_portfolio(SAMPLE)
    prices = {s: market[s]["price"] for s in ("MSFT", "GM")}
    ok, note = sz.funds_check(5000.0, portfolio, prices, DAY)
    assert note.endswith("; no price for TSLA, not counted") and f"${49 * prices['MSFT'] + 59 * prices['GM']:,.2f} could be raised" in note


# ---------------------------------------------------------------- the settings file

def test_load_config_keeps_what_it_read_until_the_file_changes(tmp_path):
    path = settings_file(tmp_path)
    assert amc.load_config(path) is amc.load_config(path) and amc.load_config() is amc.load_config()
    before = amc.load_config(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["sizing"]["float_guard"] = 1e-8
    path.write_text(json.dumps(data), encoding="utf-8")
    later = path.stat().st_mtime_ns + 5_000_000_000
    os.utime(path, ns=(later, later))                                                           # the file moves on, even on a clock too coarse to tell
    assert amc.load_config(path).sizing.float_guard == 1e-8 and amc.load_config(path) is not before


@pytest.mark.parametrize("edit, message", [
    (lambda d: d.update(extra=1), "extra"),
    (lambda d: d["sizing"].update(float_guard_x=0.05), "float_guard_x"),                  # a typo is an error, not a default
    (lambda d: d["sizing"].pop("float_guard"), "float_guard"),
    (lambda d: d.pop("sizing"), "sizing"),
    (lambda d: d.pop("config_version"), "config_version"),
    (lambda d: d["sizing"].update(float_guard=0), "float_guard"),
    (lambda d: d["sizing"].update(float_guard=0.5), "float_guard"),
])
def test_the_settings_file_rejects_what_does_not_match(tmp_path, edit, message):
    data = json.loads(amc.CONFIG_PATH.read_text(encoding="utf-8"))
    edit(data)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        amc.load_config(path)


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
    assert own <= {"prometheus.agents.account_manager.config", "prometheus.agents.account_manager.config.AccountManagerConfig", "prometheus.agents.account_manager.config.load_config",
                   "prometheus.common.portfolio", "prometheus.common.portfolio.Portfolio", "prometheus.common.portfolio.trade_amount", "prometheus.tools", "prometheus.tools.trading_math"}, own


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
