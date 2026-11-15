"""The Account Manager's sizing (`prometheus/agents/account_manager/sizing.py`, `configs/account_manager_v1.json`, `UserProfile.sizing_limits()`): how many shares to buy, from the caps of the user's limits.
Design/sizing_in_account_manager_spec.md and design choices entry 80: the Account Manager sizes (`decisions.decide_trades` calls `sizing.size_purchases`), the Critic and the Advisor only deal with recommendations.
The three allocations are tested here: the default (as much as the limits allow, in rank order), the even split (equal dollars) and the round robin (a share of each purchase in turn, in rank order: Auto mode's way).

REAL DATA ONLY. The prices are the real closes of real symbols on 2026-11-13 (PG, MSFT, TSLA, GM, XOM, EOG), read from the real stored daily bars; each average daily dollar volume (`adv_20d_dollars`) and each
stop distance (the Advisor's own reversion multiple of `atr_14`) is worked out from those same bars with the Data Clerk's own functions (`technicals.adv_20d_dollars`, `technicals.atr_14`), over the same 291 sessions
`DataClerk.request_metrics` reads. The holdings are those of the real sample portfolio `configs/portfolios/jacob_pyrett.json`; the recommendations in the faithfulness test and the session test are the real ones of
`tests/fixtures/session_real/bob_jones_oil.json` (a real run, with the Critic's real review of each, as the Critic wrote it before it stopped sizing; the capture was migrated one time to take out the fields the Critic and the Advisor no longer have, see tests/test_session_replay.py: every score, price and decision is as recorded). The profile limits are the profile's own defaults (10% in one stock, 1% lost on a stop-out, 30% added by one run) on a
$15,000 account, which is a limit of the user and not stock data; a test that needs ANOTHER limit to make one cap the tightest says so by changing that limit, never a price. No agent and no model is replaced by a
stand-in: sizing has none. The tests are skipped when the stored bars are not on the machine.

Arithmetic is checked by hand: each expected number is written in the test from the real price and the limit, not taken from the code under test.
"""
import ast
import json
import math
import os
from datetime import date, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from prometheus.agents.account_manager import config as amc
from prometheus.agents.account_manager import sizing as sz
from prometheus.agents.advisor.config import load_config as load_advisor_config
from prometheus.agents.data_clerk import technicals
from prometheus.agents.data_clerk.bars import Bar
from prometheus.common.portfolio import Portfolio, load_portfolio, trade_amount
from prometheus.common.user_profile import SizingLimits, UserProfile
from prometheus.rag.config import REPO_ROOT
from prometheus.tools.data_clerk import DataClerkTools

DAY = date(2026, 11, 13)
SYMBOLS = ["PG", "MSFT", "TSLA", "GM", "XOM", "EOG"]
EQUITY, CASH = 15000.0, 15000.0
SAMPLE = REPO_ROOT / "configs" / "portfolios" / "jacob_pyrett.json"
CAPTURE = Path(__file__).parent / "fixtures" / "session_real" / "bob_jones_oil.json"
CONFIG = amc.load_config()


@pytest.fixture(scope="module")
def market():
    """{symbol: {price, atr, stop, adv}} from the real stored bars of 2026-11-13: the close, Wilder's atr_14, the stop (the Advisor's reversion stop multiple of atr_14) and the 20-day average dollar volume."""
    try:
        found = DataClerkTools().daily_bars(SYMBOLS, DAY - timedelta(days=420), DAY)["bars"]
    except FileNotFoundError:
        pytest.skip("the stored daily bars are not on this machine")
    stop_atr = load_advisor_config().trade.stop_atr["reversion"]
    out = {}
    for symbol in SYMBOLS:
        rows = found.get(symbol) or []
        if not rows or rows[-1]["date"] != DAY.isoformat():
            pytest.skip(f"the stored bars do not hold {DAY} for {symbol}")
        bars = [Bar(date.fromisoformat(r["date"]), r["open"], r["high"], r["low"], r["close"], r["volume"], 0) for r in rows]
        atr, adv = technicals.atr_14(bars), technicals.adv_20d_dollars(bars)
        if atr is None or adv is None:
            pytest.skip(f"the stored bars of {symbol} are too few for atr_14 and adv_20d_dollars")
        out[symbol] = {"price": bars[-1].close, "atr": atr, "stop": stop_atr * atr, "adv": adv}
    return out


def limits(**changes) -> SizingLimits:
    """The limits of a moderate profile; a keyword changes one of the profile's own limits."""
    return UserProfile(user="Jacob Pyrett", risk_tolerance="moderate", **changes).sizing_limits()


def buy(symbol, market, rank=0, **changes) -> sz.Purchase:
    m = market[symbol]
    return sz.Purchase(**{"record_id": f"rec-{symbol}", "symbol": symbol, "action": "open", "price": m["price"], "stop_distance": m["stop"], "held_shares": 0, "adv_20d_dollars": m["adv"], "rank": rank, **changes})


def size(purchases, equity=EQUITY, cash=CASH, lim=None, config=CONFIG, **options) -> dict[str, sz.Sized]:
    """{record_id: Sized}"""
    return {s.record_id: s for s in sz.size_purchases(purchases, equity, cash, lim or limits(), config, **options)}


def cost(shares, price) -> float:
    return round(shares * price, 2)


def settings_file(tmp_path, **sizing) -> Path:
    """The shipped settings file with some of its `sizing` numbers changed (a threshold, not stock data)."""
    data = json.loads(amc.CONFIG_PATH.read_text(encoding="utf-8"))
    data["sizing"].update(sizing)
    path = tmp_path / "account_manager.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------- each cap alone is the one that binds, with the arithmetic by hand

def test_the_position_limit_alone_binds_one_tenth_of_the_account_in_whole_shares(market):
    p = market["PG"]["price"]
    s = size([buy("PG", market)])["rec-PG"]
    shares = math.floor(0.10 * EQUITY / p)                                  # 1,500 dollars over the price
    assert s.shares == shares >= 1 and s.binding == "position" and s.caps["position"] == shares and s.est_cost == pytest.approx(shares * p, abs=0.005)
    assert shares * p <= 0.10 * EQUITY < (shares + 1) * p                    # whole shares, the most that fit
    assert all(n > shares for name, n in s.caps.items() if name != "position" and n is not None) and s.caps["order"] is None and s.caps["budget"] is None and s.notes == []


def test_the_loss_limit_alone_binds_what_a_stop_out_may_lose_over_the_stop_distance(market):
    stop = market["GM"]["stop"]
    s = size([buy("GM", market)], lim=limits(max_loss_per_trade_pct=0.002))["rec-GM"]        # a stricter limit of the profile: 0.2% of $15,000 is $30 at risk
    shares = math.floor(0.002 * EQUITY / stop)
    assert 1 <= shares < math.floor(0.10 * EQUITY / market["GM"]["price"])
    assert (s.shares, s.binding, s.caps["loss"]) == (shares, "loss", shares) and shares * stop <= 30.0 < (shares + 1) * stop


def test_the_order_limit_alone_binds_the_dollars_of_one_order_over_the_price(market):
    p = market["GM"]["price"]
    s = size([buy("GM", market)], lim=limits(max_order_dollars=500))["rec-GM"]
    shares = math.floor(500 / p)
    assert 1 <= shares < math.floor(0.10 * EQUITY / p)
    assert (s.shares, s.binding, s.caps["order"]) == (shares, "order", shares)


def test_the_liquidity_cap_alone_binds_a_share_of_the_average_daily_dollars_over_the_price(market, tmp_path):
    m = market["GM"]
    config = amc.load_config(settings_file(tmp_path, max_adv_pct=1e-6))      # a thin share of the volume: a purchase may be one millionth of a day's dollars
    s = size([buy("GM", market)], config=config)["rec-GM"]
    shares = math.floor(1e-6 * m["adv"] / m["price"])
    assert 1 <= shares < math.floor(0.10 * EQUITY / m["price"])
    assert (s.shares, s.binding, s.caps["liquidity"]) == (shares, "liquidity", shares)


def test_the_shipped_liquidity_share_is_five_percent_of_the_average_daily_dollars(market):
    m = market["GM"]
    assert CONFIG.sizing.max_adv_pct == 0.05
    assert size([buy("GM", market)])["rec-GM"].caps["liquidity"] == math.floor(0.05 * m["adv"] / m["price"])


def test_the_money_alone_binds_the_cash_over_the_price(market):
    p = market["GM"]["price"]
    s = size([buy("GM", market)], cash=500.0)["rec-GM"]
    shares = math.floor(500.0 / p)
    assert 1 <= shares < math.floor(0.10 * EQUITY / p)
    assert (s.shares, s.binding, s.caps["money"]) == (shares, "money", shares) and s.est_cost <= 500.0


def test_the_exposure_limit_is_shared_in_rank_order_and_cuts_the_last_purchase(market):
    lim = limits(max_new_exposure_pct=0.25)                                   # 25% of $15,000 for all the purchases of one run: 3,750 dollars
    ps = [buy("PG", market, rank=1), buy("GM", market, rank=2), buy("TSLA", market, rank=3)]
    got = size(ps, lim=lim)
    pg, gm = math.floor(1500 / market["PG"]["price"]), math.floor(1500 / market["GM"]["price"])          # the first two are cut by their position limit
    left = 3750 - cost(pg, market["PG"]["price"]) - cost(gm, market["GM"]["price"])
    tsla = math.floor(left / market["TSLA"]["price"])
    assert 1 <= tsla < math.floor(1500 / market["TSLA"]["price"])             # the scenario is what it says: the exposure leaves TSLA fewer shares than its position limit does
    assert [(got[i].shares, got[i].binding) for i in ("rec-PG", "rec-GM", "rec-TSLA")] == [(pg, "position"), (gm, "position"), (tsla, "exposure")]
    assert got["rec-PG"].caps["exposure"] == math.floor(3750 / market["PG"]["price"]) and got["rec-TSLA"].caps["exposure"] == tsla
    assert sum(s.est_cost for s in got.values()) <= 3750


def test_the_rank_decides_who_gets_the_exposure_not_the_order_they_are_given_in(market):
    lim = limits(max_new_exposure_pct=0.25)
    ps = [buy("PG", market, rank=3), buy("GM", market, rank=2), buy("TSLA", market, rank=1)]       # TSLA is ranked first now, and is given last
    got = size(ps, lim=lim)
    tsla = math.floor(1500 / market["TSLA"]["price"])
    left_for_gm = 3750 - cost(tsla, market["TSLA"]["price"])                  # TSLA first, then GM (rank 2), then PG (rank 3)
    gm = min(math.floor(1500 / market["GM"]["price"]), math.floor(left_for_gm / market["GM"]["price"]))
    left_for_pg = left_for_gm - cost(gm, market["GM"]["price"])
    pg = min(math.floor(1500 / market["PG"]["price"]), math.floor(left_for_pg / market["PG"]["price"]))
    assert (got["rec-TSLA"].shares, got["rec-GM"].shares, got["rec-PG"].shares) == (tsla, gm, pg)
    assert [s.record_id for s in sz.size_purchases(ps, EQUITY, CASH, limits(), CONFIG)] == ["rec-PG", "rec-GM", "rec-TSLA"]       # the answer comes back in the order given


def test_the_cash_is_not_reserved_every_purchase_sees_all_of_it(market):
    """Jacob's rule of 2026-11-14: any available funds may be used. No purchase's cost is taken off the cash another purchase sees; the exposure limit (a limit of the profile) is what is shared."""
    ps = [buy("PG", market, rank=1), buy("GM", market, rank=2)]
    pg = math.floor(1500 / market["PG"]["price"])
    cash = cost(pg, market["PG"]["price"]) + 100.0                             # what the first buys and a hundred dollars
    got = size(ps, cash=cash)
    gm = min(math.floor(1500 / market["GM"]["price"]), math.floor(cash / market["GM"]["price"]))       # GM is cut by ALL the cash, not by what is left after PG: 16 shares, where "what is left" is 1
    assert got["rec-PG"].shares == pg and got["rec-GM"].shares == gm >= 2 and got["rec-GM"].binding == "money"
    assert got["rec-PG"].caps["money"] == math.floor(cash / market["PG"]["price"]) and got["rec-GM"].caps["money"] == math.floor(cash / market["GM"]["price"])
    assert got["rec-GM"].caps["money"] > math.floor(100.0 / market["GM"]["price"])                      # not what a running cash would have left


def test_the_money_cap_of_a_purchase_does_not_depend_on_the_other_purchases(market):
    lone = size([buy("GM", market)], cash=500.0)["rec-GM"]
    with_others = size([buy("PG", market, rank=1), buy("TSLA", market, rank=2), buy("GM", market, rank=3)], cash=500.0)
    assert with_others["rec-GM"].caps["money"] == lone.caps["money"] == math.floor(500.0 / market["GM"]["price"])
    assert all(s.caps["money"] == math.floor(500.0 / market[s.symbol]["price"]) for s in with_others.values())


def test_a_purchase_that_the_money_cannot_buy_one_share_of_is_not_made_and_says_why(market):
    s = size([buy("TSLA", market)], cash=market["TSLA"]["price"] - 1)["rec-TSLA"]
    assert (s.shares, s.est_cost, s.binding, s.caps["money"]) == (0, 0.0, "no_room", 0)
    assert "not even one share fits" in s.notes[-1] and "the tightest is money" in s.notes[-1]


def test_a_budget_caps_the_dollars_of_all_the_purchases_together(market):
    ps = [buy("PG", market, rank=1), buy("GM", market, rank=2)]
    got = size(ps, budget=1000.0)
    pg = math.floor(1000.0 / market["PG"]["price"])
    gm = math.floor((1000.0 - cost(pg, market["PG"]["price"])) / market["GM"]["price"])
    assert 1 <= pg < math.floor(1500 / market["PG"]["price"]) and gm >= 1
    assert (got["rec-PG"].shares, got["rec-PG"].binding, got["rec-GM"].shares, got["rec-GM"].binding) == (pg, "budget", gm, "budget")
    assert sum(s.est_cost for s in got.values()) <= 1000.0 and got["rec-PG"].caps["budget"] == pg


def test_a_tie_between_caps_goes_to_the_first_of_the_vocabulary(market):
    p = market["GM"]["price"]
    shares = math.floor(0.10 * EQUITY / p)
    s = size([buy("GM", market)], lim=limits(max_order_dollars=shares * p))["rec-GM"]       # the order limit allows exactly the shares the position limit does
    assert s.caps["order"] == s.caps["position"] == shares and s.binding == "position"


# ---------------------------------------------------------------- an add counts what is already held

def test_an_add_counts_the_value_already_held_against_the_position_limit(market):
    p = market["GM"]["price"]
    opened, added = size([buy("GM", market)])["rec-GM"], size([buy("GM", market, action="add", held_shares=5)])["rec-GM"]
    assert opened.caps["position"] == math.floor(1500 / p)
    assert added.caps["position"] == math.floor((1500 - 5 * p) / p) == opened.caps["position"] - 5 and added.shares == added.caps["position"] and added.binding == "position"


def test_an_add_to_a_holding_at_or_over_the_limit_buys_nothing(market):
    p = market["GM"]["price"]
    s = size([buy("GM", market, action="add", held_shares=math.ceil(1500 / p) + 1)])["rec-GM"]
    assert (s.shares, s.binding, s.caps["position"]) == (0, "no_room", 0) and "not even one share fits" in s.notes[-1]


def test_an_add_to_the_real_sample_holding_is_sized_on_the_equity_of_the_whole_portfolio(market):
    portfolio = load_portfolio(SAMPLE)
    prices = {s: market[s]["price"] for s in ("MSFT", "TSLA", "GM")}
    equity, basis = sz.equity_of(portfolio, prices)
    held = portfolio.position("GM").shares                                     # 60 shares of GM, Jacob's
    assert equity == pytest.approx(51.23 + 50 * prices["MSFT"] + 123 * prices["TSLA"] + 60 * prices["GM"]) and basis == "cash and holdings at the prices given"
    p = prices["GM"]
    rich = size([buy("GM", market, action="add", held_shares=held)], equity=equity, cash=5000.0)["rec-GM"]
    assert rich.caps["position"] == math.floor((0.10 * equity - held * p) / p) and rich.shares == rich.caps["position"] and rich.binding == "position"
    poor = size([buy("GM", market, action="add", held_shares=held)], equity=equity, cash=portfolio.cash_avail)["rec-GM"]   # the sample's own cash, $51.23, buys no GM share
    assert (poor.shares, poor.binding, poor.caps["money"]) == (0, "no_room", math.floor(51.23 / p)) and poor.caps["money"] == 0


# ---------------------------------------------------------------- the even split

def test_an_even_split_gives_equal_dollars_within_one_share_never_above_a_cap(market):
    ps = [buy("PG", market, rank=1), buy("GM", market, rank=2), buy("TSLA", market, rank=3)]
    alone = size(ps)
    even = size(ps, allocation="even")
    smallest = min(alone[i].shares * market[i[4:]]["price"] for i in alone)    # the smallest approval in dollars is the target: TSLA's three shares
    prices = {i: market[i[4:]]["price"] for i in alone}
    for i, s in even.items():
        assert s.shares == math.floor(smallest / prices[i] + 1e-9) and 1 <= s.shares <= alone[i].shares                     # never above what its own caps allow
        assert smallest - s.est_cost < prices[i] and s.est_cost <= smallest + 0.005                                          # under the target by less than one of its shares
    assert max(s.est_cost for s in even.values()) - min(s.est_cost for s in even.values()) < max(prices.values())            # equal dollars, to within one share of the dearest
    assert sum(s.est_cost for s in even.values()) <= CASH
    assert even["rec-TSLA"].binding == "position" and even["rec-PG"].binding == even["rec-GM"].binding == "even_split"       # TSLA's own limit set the amount; the others took less than they could
    assert all("even split" in s.notes[-1] and "3 purchases" in s.notes[-1] for s in even.values())


def test_when_the_money_sets_the_equal_amount_the_money_binds(market):
    ps = [buy("GM", market, rank=1), buy("PG", market, rank=2)]
    got = size(ps, cash=400.0, allocation="even")                                # $200 each
    assert [got[i].shares for i in ("rec-GM", "rec-PG")] == [math.floor(200 / market["GM"]["price"]), math.floor(200 / market["PG"]["price"])] == [2, 1]
    assert got["rec-GM"].binding == got["rec-PG"].binding == "money" and sum(s.est_cost for s in got.values()) <= 400.0


def test_a_purchase_whose_even_share_buys_no_share_is_dropped_and_the_rest_are_balanced_again(market):
    ps = [buy("GM", market, rank=1), buy("PG", market, rank=2), buy("TSLA", market, rank=3)]
    got = size(ps, cash=600.0, allocation="even")                                # $200 each buys no TSLA share: it is out, and $300 each is for the other two
    assert (got["rec-TSLA"].shares, got["rec-TSLA"].binding, got["rec-TSLA"].est_cost) == (0, "no_room", 0.0)
    assert f"does not buy one share at ${market['TSLA']['price']:,.2f}" in got["rec-TSLA"].notes[-1] and "$200.00" in got["rec-TSLA"].notes[-1]
    assert [got[i].shares for i in ("rec-GM", "rec-PG")] == [math.floor(300 / market["GM"]["price"]), math.floor(300 / market["PG"]["price"])] == [3, 1]
    assert all("$300.00 for each of 2 purchases" in got[i].notes[-1] for i in ("rec-GM", "rec-PG"))


def test_a_purchase_with_no_room_of_its_own_does_not_stop_the_others_in_an_even_split(market):
    held = math.ceil(1500 / market["TSLA"]["price"]) + 1
    ps = [buy("GM", market, rank=1), buy("TSLA", market, rank=2, action="add", held_shares=held), buy("PG", market, rank=3)]
    got = size(ps, allocation="even")
    assert got["rec-TSLA"].binding == "no_room" and got["rec-TSLA"].shares == 0
    assert got["rec-GM"].shares >= 1 and got["rec-PG"].shares >= 1 and "for each of 2 purchases" in got["rec-GM"].notes[-1]


def test_nothing_is_split_when_no_purchase_can_afford_a_share(market):
    ps = [buy("TSLA", market, rank=1), buy("MSFT", market, rank=2)]
    got = size(ps, cash=100.0, allocation="even")
    assert all(s.shares == 0 and s.binding == "no_room" and s.est_cost == 0.0 for s in got.values())


def test_an_even_split_under_a_budget(market):
    ps = [buy("PG", market, rank=1), buy("GM", market, rank=2)]
    got = size(ps, allocation="even", budget=1000.0)                              # $500 each
    assert [got[i].shares for i in ("rec-PG", "rec-GM")] == [math.floor(500 / market["PG"]["price"]), math.floor(500 / market["GM"]["price"])] == [2, 5]
    assert got["rec-PG"].binding == got["rec-GM"].binding == "budget" and sum(s.est_cost for s in got.values()) <= 1000.0


def test_an_even_split_is_also_cut_by_the_exposure_limit(market):
    ps = [buy("PG", market, rank=1), buy("GM", market, rank=2)]
    got = size(ps, lim=limits(max_new_exposure_pct=0.10, max_position_pct=0.5), allocation="even")        # 10% of $15,000 for both: $750 each
    assert [got[i].shares for i in ("rec-PG", "rec-GM")] == [math.floor(750 / market["PG"]["price"]), math.floor(750 / market["GM"]["price"])] == [4, 8]
    assert got["rec-PG"].binding == got["rec-GM"].binding == "exposure" and sum(s.est_cost for s in got.values()) <= 1500


def test_a_whole_number_of_shares_that_the_float_makes_a_hair_short_is_still_bought(market):
    price = market["PG"]["price"]
    money = round(3 * price, 2)                                                                 # exactly three shares of PG, in cents
    assert money / price < 3 and math.floor(money / price) == 2                                 # and the division is a hair short of 3: without the guard, two shares
    for allocation in (None, "even"):
        s = size([buy("PG", market)], cash=money, allocation=allocation)["rec-PG"]
        assert s.shares == 3 and s.binding == "money" and s.est_cost == money
    s = size([buy("PG", market)], lim=limits(max_order_dollars=money))["rec-PG"]
    assert s.shares == 3 and s.binding == "order"
    s = size([buy("PG", market)], budget=money)["rec-PG"]
    assert s.shares == 3 and s.binding == "budget"


def test_the_even_split_of_the_real_session_is_what_the_live_run_did(market):
    """The real run of 2026-11-13 (Bob Jones, oil, an even split of $15,000): the Critic approved XOM and EOG, and the live run bought 8 shares of each (the Account Manager decides the size). The same two purchases through the sizing give the same shares."""
    capture = json.loads(CAPTURE.read_text(encoding="utf-8"))["order"]
    reviews = {r["record_id"]: r for r in capture["reviews"]}
    purchases = [sz.Purchase(record_id=rec["identity"]["record_id"], symbol=rec["symbol"], action=rec["action"]["action"], price=reviews[rec["identity"]["record_id"]]["timing"]["deal_price"],
                             stop_distance=rec["trade"]["stop_distance"], held_shares=rec["action"]["held_shares"], adv_20d_dollars=market[rec["symbol"]]["adv"] if rec["symbol"] in market else None, rank=rank)
                 for rank, rec in enumerate(capture["recommendations"]) if reviews[rec["identity"]["record_id"]]["approved"]]
    assert [p.symbol for p in purchases] == ["XOM", "EOG"]
    traded = {d["symbol"]: d["shares"] for d in capture["trade_decisions"] if d["will_trade"] == "yes_auto"}
    assert traded == {"XOM": 8, "EOG": 8}
    each = {s.symbol: s for s in sz.size_purchases(purchases, EQUITY, CASH, limits(), CONFIG, allocation="even")}
    assert {s: each[s].shares for s in each} == traded and each["XOM"].binding == "position" and each["EOG"].binding == "even_split"
    alone = {s.symbol: s.shares for s in sz.size_purchases(purchases, EQUITY, CASH, limits(), CONFIG)}
    assert alone == {"XOM": math.floor(1500 / reviews[purchases[0].record_id]["timing"]["deal_price"]), "EOG": math.floor(1500 / reviews[purchases[1].record_id]["timing"]["deal_price"])} == {"XOM": 8, "EOG": 9}     # the 10% position limit: what the Critic itself wrote in the live run


def test_an_even_split_records_the_equal_amount_and_how_many_shared_it(market):
    ps = [buy("PG", market, rank=1), buy("GM", market, rank=2), buy("TSLA", market, rank=3)]
    alone = size(ps)
    smallest = min(alone[i].shares * market[i[4:]]["price"] for i in alone)
    even = size(ps, allocation="even")
    assert all(s.allocation == "even" and s.of == 3 and s.target_dollars == round(smallest, 2) for s in even.values())
    assert all(s.allocation == "as_much_as_possible" and s.target_dollars is None and s.of is None for s in alone.values())
    dropped = size([buy("GM", market, rank=1), buy("PG", market, rank=2), buy("TSLA", market, rank=3)], cash=600.0, allocation="even")
    assert dropped["rec-TSLA"].shares == 0 and dropped["rec-TSLA"].target_dollars == 200.0 and dropped["rec-TSLA"].of == 3 and dropped["rec-GM"].target_dollars == 300.0 and dropped["rec-GM"].of == 2


# ---------------------------------------------------------------- the round robin (Auto mode: Jacob, 2026-11-14)
# "Buy a share of the first stock, buy a share of the next stock, and keep buying shares until all the desired funds are used up." Each stock ends with a FINAL number of shares.

def xom_eog(market, budget, first="XOM", **options):
    """XOM and EOG (the two real approved oil stocks of the real session), `first` ranked first, a round robin of `budget` dollars over the profile's own limits and a $15,000 account."""
    second = "EOG" if first == "XOM" else "XOM"
    return size([buy(first, market, rank=1), buy(second, market, rank=2)], allocation="round_robin", budget=budget, **options)


def test_round_robin_buys_one_share_at_a_time_in_rank_order_and_the_last_share_goes_to_whoever_is_first(market):
    """A pot of about two shares of each, worked by hand from the real closes of 2026-11-13: XOM 170.59 and EOG 158.32. Round 1: XOM 170.59, EOG 328.91. Round 2: XOM 499.50 fits $500, EOG 657.82 does not.
    XOM 2 and EOG 1. With EOG ranked first the rounds go EOG 158.32, XOM 328.91, EOG 487.23 (fits), XOM 657.82 (does not): EOG 2 and XOM 1. The order of the rounds decides who gets the last share."""
    xom, eog = market["XOM"]["price"], market["EOG"]["price"]
    assert (xom, eog) == (170.59, 158.32)                                                            # the real closes: the hand arithmetic below is for these
    assert 2 * xom + eog <= 500 < 2 * xom + 2 * eog and 2 * eog + xom <= 500 < 2 * eog + 2 * xom                                     # the scenario is what it says, in either order
    a = xom_eog(market, 500.0, first="XOM")
    assert (a["rec-XOM"].shares, a["rec-EOG"].shares) == (2, 1) and (a["rec-XOM"].est_cost, a["rec-EOG"].est_cost) == (341.18, 158.32) and 341.18 + 158.32 == 499.5
    b = xom_eog(market, 500.0, first="EOG")
    assert (b["rec-EOG"].shares, b["rec-XOM"].shares) == (2, 1) and (b["rec-EOG"].est_cost, b["rec-XOM"].est_cost) == (316.64, 170.59) and 316.64 + 170.59 == 487.23
    assert all(s.binding == "budget" and s.allocation == "round_robin" for s in (*a.values(), *b.values()))          # the $500 asked for is what closed every purchase


def test_round_robin_notes_say_how_many_rounds_and_why_each_purchase_closed(market):
    got = xom_eog(market, 500.0)
    assert all(s.notes[0] == "round robin: 2 rounds, one share of each purchase in rank order, $500.00 to spend (budget), $0.50 left" for s in got.values())
    assert got["rec-XOM"].notes[1] == "closed at 2 shares: the budget has no money for another share" and got["rec-EOG"].notes[1] == "closed at 1 share: the budget has no money for another share"


def test_round_robin_costs_are_cents_the_way_the_account_book_rounds_them_and_the_total_is_their_sum(market):
    got = xom_eog(market, 2000.0)
    for i, s in got.items():
        assert s.est_cost == trade_amount(s.shares, market[s.symbol]["price"])
    assert sum(s.est_cost for s in got.values()) == pytest.approx(sum(round(s.shares * market[s.symbol]["price"], 2) for s in got.values()), abs=1e-9)


def test_a_stock_at_its_own_limit_stops_alone_while_the_others_go_on(market):
    """An order limit of $400 for one order: PG (169.80) allows 2 shares, GM (86.00) allows 4. Rounds 1 and 2 take a share of each; round 3 PG is at its limit and closes ('order'), GM takes a third and a fourth;
    round 5 GM is at its limit and closes too. Nobody else's limit stops a stock."""
    pg, gm = market["PG"]["price"], market["GM"]["price"]
    assert (math.floor(400 / pg), math.floor(400 / gm)) == (2, 4)
    got = size([buy("PG", market, rank=1), buy("GM", market, rank=2)], lim=limits(max_order_dollars=400), allocation="round_robin")
    assert (got["rec-PG"].shares, got["rec-PG"].binding, got["rec-GM"].shares, got["rec-GM"].binding) == (2, "order", 4, "order")
    assert got["rec-PG"].notes[1] == "closed at 2 shares: it is at its order limit" and "5 rounds" not in got["rec-PG"].notes[0] and "4 rounds" in got["rec-PG"].notes[0]
    assert got["rec-PG"].est_cost == trade_amount(2, pg) and got["rec-GM"].est_cost == trade_amount(4, gm)


def test_the_pot_is_used_to_less_than_the_cheapest_share_that_could_still_have_been_bought(market):
    """GM (86.00) ranked first, EOG (158.32), XOM (170.59), a pot of $1,000. Round 1 spends 414.91, round 2 829.82, round 3: GM 915.82 fits, EOG 1,074.14 and XOM do not. Round 4: GM 1,001.82 does not.
    GM 3, EOG 2, XOM 2, $84.18 left: under the 86.00 of GM's next share."""
    gm, eog, xom = market["GM"]["price"], market["EOG"]["price"], market["XOM"]["price"]
    got = size([buy("GM", market, rank=1), buy("EOG", market, rank=2), buy("XOM", market, rank=3)], allocation="round_robin", budget=1000.0)
    assert [got[i].shares for i in ("rec-GM", "rec-EOG", "rec-XOM")] == [3, 2, 2] and all(s.binding == "budget" for s in got.values())
    spent = sum(s.est_cost for s in got.values())
    assert spent == pytest.approx(3 * gm + 2 * eog + 2 * xom) == pytest.approx(915.82) and spent <= 1000.0 and 1000.0 - spent < min(gm, eog, xom)


def test_the_round_robin_differs_from_the_even_split_and_from_the_default_on_the_same_inputs(market):
    ps = [buy("GM", market, rank=1), buy("EOG", market, rank=2), buy("XOM", market, rank=3)]
    robin, even, greedy = (size(ps, budget=1000.0, allocation=a) for a in ("round_robin", "even", None))
    shares = lambda got: [got[i].shares for i in ("rec-GM", "rec-EOG", "rec-XOM")]
    assert shares(robin) == [3, 2, 2] and shares(even) == [3, 2, 1] and shares(greedy) == [11, 0, 0]
    assert even["rec-EOG"].binding == "budget" and greedy["rec-GM"].binding == "budget" and [greedy[i].binding for i in ("rec-EOG", "rec-XOM")] == ["no_room", "no_room"]     # the default gives all of the pot to the first


def test_the_exposure_pool_the_purchases_share_closes_them_and_the_room_left_is_under_a_share(market):
    """5% of $15,000 for all the purchases of the run: $750. XOM 170.59 and EOG 158.32: round 1 spends 170.59 then 328.91, round 2 499.50 then 657.82; round 3: XOM would make 828.41 and EOG 816.14, over $750.
    XOM 2, EOG 2, both closed by 'exposure', and the $92.18 of room left is under the price of either."""
    xom, eog = market["XOM"]["price"], market["EOG"]["price"]
    got = size([buy("XOM", market, rank=1), buy("EOG", market, rank=2)], lim=limits(max_new_exposure_pct=0.05, max_position_pct=0.5), allocation="round_robin")
    assert (got["rec-XOM"].shares, got["rec-EOG"].shares) == (2, 2) and all(s.binding == "exposure" for s in got.values())
    assert 2 * xom + 2 * eog == pytest.approx(657.82) and 750 - (2 * xom + 2 * eog) < min(xom, eog) and 3 * xom + 2 * eog > 750
    greedy = size([buy("XOM", market, rank=1), buy("EOG", market, rank=2)], lim=limits(max_new_exposure_pct=0.05, max_position_pct=0.5))
    assert (greedy["rec-XOM"].shares, greedy["rec-EOG"].shares, greedy["rec-EOG"].binding) == (4, 0, "no_room")                   # the default gives the whole room to the first and the second gets none


def test_a_budget_above_the_cash_is_the_cash_and_no_budget_is_the_cash(market):
    xom, eog = market["XOM"]["price"], market["EOG"]["price"]
    for options in ({"budget": 20000.0}, {}):
        got = size([buy("XOM", market, rank=1), buy("EOG", market, rank=2)], cash=500.0, allocation="round_robin", **options)
        assert (got["rec-XOM"].shares, got["rec-EOG"].shares) == (2, 1) and all(s.binding == "money" for s in got.values()) and sum(s.est_cost for s in got.values()) <= 500.0
        assert got["rec-XOM"].notes[0].endswith("$500.00 to spend (money), $0.50 left")
        assert got["rec-XOM"].caps["money"] == math.floor(500 / xom) and got["rec-EOG"].caps["money"] == math.floor(500 / eog)     # the cash is not reserved: each purchase still shows all of it


def test_round_robin_caps_show_the_whole_room_as_the_even_split_does(market):
    got = xom_eog(market, 500.0)
    for symbol in ("XOM", "EOG"):
        s = got[f"rec-{symbol}"]
        assert s.caps["budget"] == math.floor(500 / market[symbol]["price"]) and s.caps["exposure"] == math.floor(0.30 * EQUITY / market[symbol]["price"]) and s.caps["money"] == math.floor(CASH / market[symbol]["price"])
        assert s.caps["position"] == math.floor(1500 / market[symbol]["price"]) and list(s.caps) == list(sz.CAP_NAMES)


def test_a_purchase_the_pot_cannot_buy_one_share_of_has_no_room_and_the_others_still_go_on(market):
    """XOM (170.59) is ranked first and a pot of $100 buys none of it; GM (86.00) is behind it and takes one share. Closed by 'budget', not by a limit: the note says so."""
    got = size([buy("XOM", market, rank=1), buy("GM", market, rank=2)], allocation="round_robin", budget=100.0)
    assert (got["rec-XOM"].shares, got["rec-XOM"].est_cost, got["rec-XOM"].binding) == (0, 0.0, "no_room") and "not even one share fits: the budget has no money for another share" in got["rec-XOM"].notes[-1]
    assert (got["rec-GM"].shares, got["rec-GM"].binding, got["rec-GM"].est_cost) == (1, "budget", 86.0)


def test_round_robin_is_in_rank_order_not_in_the_order_given_and_ties_go_in_the_order_given(market):
    xom, eog = market["XOM"]["price"], market["EOG"]["price"]
    ps = [buy("EOG", market, rank=2), buy("XOM", market, rank=1)]                                           # EOG is given first and ranked second
    got = size(ps, allocation="round_robin", budget=500.0)
    assert (got["rec-XOM"].shares, got["rec-EOG"].shares) == (2, 1) and [s.record_id for s in sz.size_purchases(ps, EQUITY, CASH, limits(), CONFIG, allocation="round_robin", budget=500.0)] == ["rec-EOG", "rec-XOM"]
    tied = size([buy("EOG", market, rank=0), buy("XOM", market, rank=0)], allocation="round_robin", budget=500.0)           # a tie: the order given decides
    assert (tied["rec-EOG"].shares, tied["rec-XOM"].shares) == (2, 1)


def test_a_purchase_that_cannot_be_sized_is_left_out_of_the_round_robin_and_does_not_stop_the_others(market):
    got = size([buy("GM", market, rank=1, price=None), buy("XOM", market, rank=2), buy("EOG", market, rank=3)], allocation="round_robin", budget=500.0)
    assert (got["rec-GM"].shares, got["rec-GM"].binding) == (0, "no_room") and "not sized: no price" in got["rec-GM"].notes[-1]
    assert (got["rec-XOM"].shares, got["rec-EOG"].shares) == (2, 1)


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


def test_round_robin_shares_are_never_above_a_purchases_own_limits(market):
    got = size([buy("XOM", market, rank=1), buy("EOG", market, rank=2)], allocation="round_robin")           # all the cash, every purchase at its own position limit
    assert (got["rec-XOM"].shares, got["rec-EOG"].shares) == (8, 9) and (got["rec-XOM"].binding, got["rec-EOG"].binding) == ("position", "position")
    assert all(s.shares <= min(n for k, n in s.caps.items() if k in sz.LIMIT_CAPS and n is not None) for s in got.values())


def test_the_round_robin_asks_nothing_of_a_purchase_that_is_a_sale_or_a_hold_and_an_empty_list_is_an_empty_answer():
    assert sz.size_purchases([], EQUITY, CASH, limits(), CONFIG, allocation="round_robin") == []


def test_the_allocations_are_the_three_names_and_round_robin_is_one_of_them():
    assert sz.ALLOCATIONS == ("even", "round_robin", "as_much_as_possible")


# ---------------------------------------------------------------- the real session's two approved purchases

def test_the_real_sessions_xom_and_eog_have_the_caps_the_live_critic_wrote_before_it_stopped_sizing():
    """The Critic of the live run of 2026-11-13 (Bob Jones, conservative, a $15,000 account, the profile's own limits) recorded these caps for the two approved purchases: XOM position 8, loss 123, exposure 26; EOG
    position 9, loss 18, exposure 19 (after XOM's 8 shares). Worked again by hand from the real record of each (deal price, plan stop distance) they are the same, now the Account Manager's."""
    order = json.loads(CAPTURE.read_text(encoding="utf-8"))["order"]
    reviews = {r["record_id"]: r for r in order["reviews"]}
    approved = [rec for rec in order["recommendations"] if reviews[rec["identity"]["record_id"]]["approved"]]
    assert [rec["symbol"] for rec in approved] == ["XOM", "EOG"]
    purchases = [sz.Purchase(record_id=rec["identity"]["record_id"], symbol=rec["symbol"], action="open", price=reviews[rec["identity"]["record_id"]]["timing"]["deal_price"], stop_distance=rec["trade"]["stop_distance"],
                             adv_20d_dollars=rec["risk"]["adv_20d_dollars"], rank=rec["ranking"]["rank"]) for rec in approved]
    xom, eog = purchases
    assert (xom.price, xom.stop_distance, eog.price, eog.stop_distance) == (170.59, 1.2114, 158.32, 8.292)
    got = sz.size_purchases(purchases, EQUITY, CASH, limits(), CONFIG)
    assert (got[0].caps["position"], got[0].caps["loss"], got[0].caps["exposure"]) == (math.floor(1500 / 170.59), math.floor(150 / 1.2114), math.floor(4500 / 170.59)) == (8, 123, 26)
    assert (got[1].caps["position"], got[1].caps["loss"], got[1].caps["exposure"]) == (math.floor(1500 / 158.32), math.floor(150 / 8.292), math.floor((4500 - 8 * 170.59) / 158.32)) == (9, 18, 19)
    assert [(s.shares, s.binding) for s in got] == [(8, "position"), (9, "position")]


# ---------------------------------------------------------------- what cannot be worked out is None and is said

def test_a_missing_stop_distance_leaves_the_loss_cap_none_and_the_others_apply(market):
    s = size([buy("GM", market, stop_distance=None)])["rec-GM"]
    assert s.caps["loss"] is None and s.caps["position"] == math.floor(1500 / market["GM"]["price"]) == s.shares and s.binding == "position"
    assert s.notes == ["loss cap not worked out: the plan has no stop distance"]


def test_a_missing_volume_leaves_the_liquidity_cap_none_and_says_so(market):
    s = size([buy("GM", market, adv_20d_dollars=None)])["rec-GM"]
    assert s.caps["liquidity"] is None and s.shares == s.caps["position"] >= 1
    assert s.notes == ["liquidity cap not worked out: the average daily dollar volume is not known"]


def test_an_unknown_equity_leaves_the_position_loss_and_exposure_caps_none_and_the_order_limit_sizes_it(market):
    p = market["GM"]["price"]
    s = size([buy("GM", market)], equity=None, lim=limits(max_order_dollars=500))["rec-GM"]
    assert (s.caps["position"], s.caps["loss"], s.caps["exposure"]) == (None, None, None) and s.shares == math.floor(500 / p) == s.caps["order"] and s.binding == "order"
    assert s.notes == ["position cap not worked out: the equity is not known", "loss cap not worked out: the equity is not known", "exposure cap not worked out: the equity is not known"]


def test_with_no_equity_and_no_stop_the_loss_note_gives_both_reasons(market):
    s = size([buy("GM", market, stop_distance=None)], equity=None)["rec-GM"]
    assert "loss cap not worked out: the equity is not known and the plan has no stop distance" in s.notes


def test_no_price_means_no_size(market):
    for price in (None, 0.0, -1.0, float("nan")):
        s = size([buy("GM", market, price=price)])["rec-GM"]
        assert (s.shares, s.est_cost, s.binding) == (0, 0.0, "no_room") and set(s.caps) == set(sz.CAP_NAMES) and s.caps["money"] is None
        assert s.notes[0] == "GM has no price, so no cap can be worked out" and "not sized: no price" in s.notes[-1]


def test_when_no_limit_at_all_can_be_worked_out_the_purchase_is_not_sized_and_never_gets_all_the_cash(market):
    s = size([buy("GM", market, stop_distance=None, adv_20d_dollars=None)], equity=None)["rec-GM"]     # no equity, no stop, no volume, and the profile sets no order limit
    assert (s.shares, s.est_cost, s.binding) == (0, 0.0, "no_room") and s.caps["money"] is None
    assert "no limit of the user's could be worked out" in s.notes[-1] and "never computed from a guess" in s.notes[-1]


def test_a_volume_of_zero_is_a_real_zero_and_leaves_no_room(market):
    s = size([buy("GM", market, adv_20d_dollars=0.0)])["rec-GM"]
    assert (s.caps["liquidity"], s.shares, s.binding) == (0, 0, "no_room")


def test_every_sized_has_every_cap_name_and_a_whole_number_of_shares(market):
    for s in size([buy("PG", market, rank=1), buy("GM", market, rank=2, stop_distance=None)], budget=2000.0).values():
        assert list(s.caps) == list(sz.CAP_NAMES) and isinstance(s.shares, int) and s.binding in sz.BINDINGS


# ---------------------------------------------------------------- the limits a user raised for one purchase

def test_a_limit_the_user_raised_for_this_purchase_replaces_the_profile_for_this_call_only(market):
    p = market["PG"]["price"]
    profile = limits()
    plain = size([buy("PG", market)], lim=profile)["rec-PG"]
    raised = size([buy("PG", market)], lim=profile, limit_overrides={"max_position_pct": 0.20})["rec-PG"]
    assert plain.shares == math.floor(1500 / p) and raised.shares == math.floor(3000 / p) > plain.shares and raised.binding == "position"
    assert raised.notes == ["max_position_pct is 0.2 for this purchase (the profile says 0.1): the user changed it"]
    assert profile.max_position_pct == 0.10 and size([buy("PG", market)], lim=profile)["rec-PG"] == plain                       # the profile is as it was


def test_an_order_limit_can_be_set_or_taken_off_for_one_call(market):
    p = market["GM"]["price"]
    capped = size([buy("GM", market)], lim=limits(max_order_dollars=500))["rec-GM"]
    freed = size([buy("GM", market)], lim=limits(max_order_dollars=500), limit_overrides={"max_order_dollars": None})["rec-GM"]
    assert capped.shares == math.floor(500 / p) and freed.shares == math.floor(1500 / p) and freed.caps["order"] is None
    assert freed.notes == ["max_order_dollars is no limit for this purchase (the profile says 500): the user changed it"]


def test_an_override_that_is_not_a_limit_or_not_valid_is_refused(market):
    for bad in ({"max_position_pct": 1.5}, {"max_position_pct": 0}, {"max_loss_per_trade_pct": -0.01}, {"max_position": 0.2}, {"budget": 100}):
        with pytest.raises(ValueError):
            size([buy("GM", market)], limit_overrides=bad)


def test_an_override_equal_to_the_profile_says_nothing(market):
    assert size([buy("GM", market)], limit_overrides={"max_position_pct": 0.10})["rec-GM"].notes == []


# ---------------------------------------------------------------- what only a program can get wrong is an error

def test_arguments_that_make_no_sense_are_refused(market):
    one = [buy("GM", market)]
    for options in ({"allocation": "random"}, {"budget": 0}, {"budget": -5.0}, {"budget": True}, {"budget": float("nan")}, {"budget": float("inf")}, {"budget": "100"}, {"cash": -1.0}, {"cash": float("nan")}):
        with pytest.raises(ValueError):
            size(one, **options)
    with pytest.raises(ValueError, match="same record_id"):
        size([buy("GM", market), buy("GM", market)])


def test_both_names_of_the_default_allocation_are_the_same(market):
    ps = [buy("PG", market, rank=1), buy("GM", market, rank=2)]
    assert size(ps, allocation="as_much_as_possible") == size(ps, allocation=None) == size(ps)


def test_no_purchases_is_an_empty_answer():
    assert sz.size_purchases([], EQUITY, CASH, limits(), CONFIG) == []


def test_the_records_refuse_a_field_they_do_not_know(market):
    with pytest.raises(ValidationError):
        sz.Purchase(record_id="a", symbol="GM", action="open", price=1.0, colour="red")
    with pytest.raises(ValidationError):
        sz.Purchase(record_id="a", symbol="GM", action="reduce", price=1.0)                    # a sale is not sized here
    with pytest.raises(ValidationError):
        sz.Purchase(record_id="a", symbol="GM", action="open", price=1.0, held_shares=-1)
    got = size([buy("GM", market)])["rec-GM"]
    with pytest.raises(ValidationError):
        sz.Sized(**{**got.model_dump(), "colour": "red"})
    with pytest.raises(ValidationError):
        got.shares = 5                                                                          # frozen


# ---------------------------------------------------------------- equity

def test_equity_is_the_cash_and_the_holdings_at_their_prices(market):
    portfolio = load_portfolio(SAMPLE)
    prices = {s: market[s]["price"] for s in ("MSFT", "TSLA", "GM")}
    equity, basis = sz.equity_of(portfolio, prices)
    assert equity == pytest.approx(51.23 + 50 * prices["MSFT"] + 123 * prices["TSLA"] + 60 * prices["GM"]) and basis == "cash and holdings at the prices given"


def test_a_holding_with_no_price_falls_back_to_its_cost_basis_and_the_note_says_so(market):
    portfolio = load_portfolio(SAMPLE)
    prices = {s: market[s]["price"] for s in ("MSFT", "GM")}                                   # no price for TSLA
    equity, basis = sz.equity_of(portfolio, prices)
    assert equity == pytest.approx(51.23 + 50 * prices["MSFT"] + 123 * 300.00 + 60 * prices["GM"])                    # TSLA at its cost basis, $300
    assert basis == "cash and holdings at the prices given; TSLA at cost basis (no price)"
    equity, basis = sz.equity_of(portfolio, {})                                                  # no price at all
    assert equity == pytest.approx(51.23 + 50 * 150.0 + 123 * 300.0 + 60 * 10.0) and basis.endswith("MSFT, TSLA, GM at cost basis (no price)")
    assert sz.equity_of(portfolio, {"MSFT": 0.0, "TSLA": float("nan"), "GM": None})[0] == equity     # a price that is not a usable number is no price


def test_an_account_of_only_cash_and_an_empty_account():
    assert sz.equity_of(Portfolio(user="Jacob Pyrett", cash_avail=15000.0), {}) == (15000.0, "cash only (no holdings)")
    assert sz.equity_of(Portfolio(user="Jacob Pyrett"), {}) == (None, "the account holds no cash and no holdings")


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


# ---------------------------------------------------------------- the profile's limits

def test_the_profile_gives_its_four_limits_to_the_sizing():
    assert UserProfile(user="Jacob Pyrett", risk_tolerance="moderate").sizing_limits() == SizingLimits(max_position_pct=0.10, max_loss_per_trade_pct=0.01, max_new_exposure_pct=0.30, max_order_dollars=None)
    profile = UserProfile(user="Jacob Pyrett", risk_tolerance="aggressive", max_position_pct=0.2, max_loss_per_trade_pct=0.02, max_new_exposure_pct=0.5, max_order_dollars=2500)
    assert profile.sizing_limits().model_dump() == {"max_position_pct": 0.2, "max_loss_per_trade_pct": 0.02, "max_new_exposure_pct": 0.5, "max_order_dollars": 2500}
    assert set(SizingLimits.model_fields) <= set(UserProfile.model_fields) and not set(SizingLimits.model_fields) & set(profile.risk_view().model_dump())         # the profile holds the four numbers; the Critic's view holds none of them (entry 80)


def test_the_limits_are_frozen_closed_and_checked():
    lim = limits()
    with pytest.raises(ValidationError):
        lim.max_position_pct = 0.5
    with pytest.raises(ValidationError):
        SizingLimits(max_position_pct=0.1, max_loss_per_trade_pct=0.01, max_new_exposure_pct=0.3, colour="red")
    with pytest.raises(ValidationError):
        SizingLimits(max_position_pct=1.5, max_loss_per_trade_pct=0.01, max_new_exposure_pct=0.3)
    assert lim.changed({"max_position_pct": 0.2}).max_position_pct == 0.2 and lim.max_position_pct == 0.10
    with pytest.raises(ValueError):
        lim.changed({"nope": 1})


# ---------------------------------------------------------------- the settings file

def test_the_shipped_file_loads_and_holds_the_advisors_liquidity_share():
    config = amc.load_config()
    assert config.config_version == "1" and config.sizing.max_adv_pct == 0.05 and config.sizing.float_guard == 1e-9


def test_load_config_keeps_what_it_read_until_the_file_changes(tmp_path):
    path = settings_file(tmp_path)
    assert amc.load_config(path) is amc.load_config(path) and amc.load_config() is amc.load_config()
    before = amc.load_config(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["sizing"]["max_adv_pct"] = 0.02
    path.write_text(json.dumps(data), encoding="utf-8")
    later = path.stat().st_mtime_ns + 5_000_000_000
    os.utime(path, ns=(later, later))                                                           # the file moves on, even on a clock too coarse to tell
    assert amc.load_config(path).sizing.max_adv_pct == 0.02 and amc.load_config(path) is not before


@pytest.mark.parametrize("edit, message", [
    (lambda d: d.update(extra=1), "extra"),
    (lambda d: d["sizing"].update(max_adv_pcnt=0.05), "max_adv_pcnt"),                     # a typo is an error, not a default
    (lambda d: d["sizing"].pop("max_adv_pct"), "max_adv_pct"),
    (lambda d: d["sizing"].pop("float_guard"), "float_guard"),
    (lambda d: d.pop("sizing"), "sizing"),
    (lambda d: d.pop("config_version"), "config_version"),
    (lambda d: d["sizing"].update(max_adv_pct="five percent"), "max_adv_pct"),
    (lambda d: d["sizing"].update(max_adv_pct=0), "max_adv_pct"),
    (lambda d: d["sizing"].update(max_adv_pct=1.5), "max_adv_pct"),
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


# ---------------------------------------------------------------- the module's own imports

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
                 "langchain", "langchain_core", "langgraph", "openai", "anthropic", "httpx", "requests", "urllib", "socket", "aiohttp", "psycopg", "psycopg2", "subprocess", "os", "dotenv")
    for name in names:
        assert not any(name == bad or name.startswith(bad + ".") for bad in forbidden), f"sizing.py imports {name}"
    own = {n for n in names if n.startswith("prometheus")}
    assert own <= {"prometheus.agents.account_manager.config", "prometheus.agents.account_manager.config.AccountManagerConfig", "prometheus.agents.account_manager.config.load_config",
                   "prometheus.common.portfolio", "prometheus.common.portfolio.Portfolio", "prometheus.common.portfolio.trade_amount",
                   "prometheus.common.user_profile", "prometheus.common.user_profile.SizingLimits", "prometheus.tools", "prometheus.tools.trading_math"}, own


def test_the_settings_loader_reads_nothing_but_the_file():
    names = imports_of(REPO_ROOT / "prometheus" / "agents" / "account_manager" / "config.py")
    assert {n for n in names if n.startswith("prometheus")} <= {"prometheus.rag.config", "prometheus.rag.config.REPO_ROOT"}
    assert not any(n.split(".")[0] in ("langchain", "langchain_core", "openai", "httpx", "requests") for n in names)
