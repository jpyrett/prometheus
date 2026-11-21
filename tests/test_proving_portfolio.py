"""`SimPortfolio`, the Time Machine's own money (design/proving_ground_spec.md sections 5 and 14; design choices entry 126; P5, P6, P7, P10).

REAL DATA ONLY. The purchases are the fills of the REAL Trader in backtest mode (no model) over two real orders of the sessions of 2026-11-13, through the real Account Manager's per-trade handshake: `bob_jones_oil` (XOM 8 and EOG 8 whole shares) and
`auto_even_split_dollars` (five orders in dollars, so fractional shares: EOG, XOM, CVX, and the neutral OXY and META, which have no plan). They fill at the open of Monday 2026-11-16. The portfolio is walked day by day over the stored bars of the finished
sessions 2026-11-13 to 2026-11-19 exactly as the engine will walk it (the day's fills, then the exits, then the mark), and every number is compared with what the Trader and the Account Manager said (the Trader's own `Trade.outcome` and the Account
Manager's own cash after the purchases) and with a reading of the bars by hand, in exact Decimal. The split case puts a real purchase of Netflix, filled at the real open of 2025-11-14, across its real 10-for-1 split of 2025-11-17.
"""
from datetime import date
from decimal import Decimal

import pytest
from proving_helpers import close_on, plans_of, real_bars, real_records, replayed, restated, stored_bars_available

from prometheus.agents.trader import simulator
from prometheus.agents.trader.record import Fill, Trade
from prometheus.common.portfolio import trade_amount
from prometheus.proving import record, splits
from prometheus.proving.portfolio import IdentityError, LotInfo, SimPortfolio

pytestmark = pytest.mark.skipif(not stored_bars_available(), reason="the stored daily bars (data/historical/day_aggs_v1) are not on this machine")

DAYS = ["2026-11-13", "2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19"]


def walk(order, cash: str, bars):
    """The engine's walk, day by day: the fills of the day, the exits, the mark. Returns the account, its points, its exits and the cash right after the purchases."""
    account = SimPortfolio(Decimal(cash), user="Bob Jones")
    info, points, exits, after_buys = plans_of(order), [], [], None
    for day in DAYS:
        bought = [t for t in order.trades if t["status"] == "filled" and t["fill"]["date"] == day]
        account.apply_fills(bought, info)
        after_buys = account.cash if bought else after_buys
        exits += account.step(date.fromisoformat(day), bars)
        points.append(account.mark(date.fromisoformat(day), bars))
    return account, points, exits, after_buys


def test_a_held_symbol_with_no_bar_keeps_its_close_and_the_point_says_so_and_a_sale_or_a_fill_the_cash_cannot_pay_is_refused():
    """2026-11-21 is a Saturday: no bar, so each open position keeps its close of the 19th and the point names it in `stale` (no price is invented, the equity does not move). `apply_fills` takes the Trader's purchases only: a sale is a ValueError,
    and a fill the cash cannot pay for breaks an identity (IdentityError, an AssertionError: a run stops on it)."""
    order = replayed("auto_even_split_dollars")
    bars = real_bars({t["symbol"] for t in order.trades})
    account, points, _, _ = walk(order, "40000.00", bars)
    saturday = account.mark(date(2026, 11, 21), bars)
    assert saturday.stale == ["META", "OXY"] and (saturday.equity, saturday.cash, saturday.invested) == (points[-1].equity, points[-1].cash, points[-1].invested)
    assert [p.stale for p in account.positions(date(2026, 11, 21))] == [True, True] and [p.stale for p in account.positions(date(2026, 11, 19))] == [False, False]
    sale = Trade.model_validate({**order.trades[0], "side": "sell"})
    with pytest.raises(ValueError, match="never sells but by the plan"):
        SimPortfolio(Decimal("40000.00")).apply_fills([sale])
    with pytest.raises(IdentityError, match="the cash is"):
        SimPortfolio(Decimal("100.00")).apply_fills(order.trades)
    assert isinstance(IdentityError("x"), AssertionError)


def test_a_position_that_crosses_a_real_split_keeps_its_value_and_its_plan_and_with_no_table_it_is_not_followed_across_it(tmp_path):
    """Netflix split 10 for 1 on 2025-11-17. A purchase of 3 shares at the real open of 2025-11-14 (the real AMD plan's distances on its real bars: see test_proving_outcomes), time stop 2026-01-08. With the table the lot becomes 30 shares on the 17th, its stop
    and cost basis a tenth, the equity does not fall by 90% and it is sold at the time stop at the real close of that day, 90.53 a share: 2,715.90. With NO table the position is not followed past the suspected split (the false stop of the raw bars is not
    taken): it stays open at its last good close, stale, and `warnings` says why."""
    bars = real_bars({"NFLX"})
    net = bars["NFLX"]
    plan = restated(record.plan_of(next(r for r in real_records() if r.record_id == "83910656bece").rec), close_on(net, "2025-11-13"), date(2025, 11, 13))
    opened = simulator.bar_price(simulator.bar_on(net, "2025-11-14"), "open")
    fill = Fill(price=opened, shares=Decimal(3), amount=trade_amount(Decimal(3), opened), date="2025-11-14", price_source="bar_open")
    trade = Trade(trade_id="T1", kind="recommendation", record_id="r1", symbol="NFLX", side="buy", shares_approved=Decimal(3), shares=Decimal(3), status="filled", mode="backtest", fill=fill)
    table = splits.SplitTable(symbol="NFLX", splits=[splits.Split(execution_date=date(2025, 11, 17), split_from=1.0, split_to=10.0)])
    days = [date.fromisoformat(b["date"]) for b in bars["SPY"] if "2025-11-14" <= b["date"] <= "2026-01-08"]
    info = {"r1": LotInfo(plan=plan, key="k", decision_day=date(2025, 11, 13))}

    account = SimPortfolio(Decimal("10000.00"), splits={"NFLX": table})
    account.apply_fills([trade], info)
    assert opened == Decimal("1142.7300") and account.cash == Decimal("6571.81") and account.lots[0].stop == plan.stop
    seen = {}
    for day in days:
        account.step(day, bars)
        seen[day.isoformat()] = account.mark(day, bars)
        if day == date(2025, 11, 14):
            before = seen[day.isoformat()]
    after = seen["2025-11-17"]
    assert abs(after.equity - before.equity) / before.equity < Decimal("0.03") and after.invested == trade_amount(Decimal(30), price_close(net, "2025-11-17"))      # a tenth of the price, ten times the shares
    assert len(account.closed) == 1 and account.lots == []
    sold = account.closed[0]
    assert (sold.date.isoformat(), sold.reason, sold.shares, sold.price, sold.amount, sold.cost, sold.gain, sold.sessions_held) == ("2026-01-08", "time_stop", Decimal(30), Decimal("90.5300"), Decimal("2715.90"), Decimal("3428.19"), Decimal("-712.29"), 36)
    assert account.cash == Decimal("9287.71") and account.realized == Decimal("-712.29") and account.warnings == []

    blind = SimPortfolio(Decimal("10000.00"))
    blind.apply_fills([trade], info)
    for day in days:
        blind.step(day, bars)
        point = blind.mark(day, bars)
    assert blind.closed == [] and len(blind.lots) == 1 and point.stale == ["NFLX"] and blind.cash == Decimal("6571.81")
    assert blind.lots[0].last_close == price_close(net, "2025-11-14") and point.invested == trade_amount(Decimal(3), price_close(net, "2025-11-14"))
    assert blind.warnings == ["NFLX moved -90.0% from one close to the next open on 2025-11-17, which looks like a split, and no split table is stored for it: the position is not followed past that day"]


def price_close(series, day: str) -> Decimal:
    return simulator.bar_price(simulator.bar_on(series, day), "close")
