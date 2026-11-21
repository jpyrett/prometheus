"""The TRAILING STOP of the Advisor's exit plan (design/advisor_exit_plan_spec.md rules E2, E3, E4, E7 and E8; design choices entry 139) on REAL data.

The TREND family has no target and a stop that rises with the highest close. Its follower is `simulator.follow(trail_distance=)` and the Proving Ground's outcome, account and Live view all call it. Every test below reads the REAL stored daily bars (read only) and
works the expected result out BY HAND in the test, with its own loop, and pins the answer.

HOW THE CASE WAS FOUND (a throwaway script, not kept): over the stored bars of 17 liquid symbols (XOM, MSFT, AAPL, NVDA, AMD, CVX, COP, PG, KO, JPM, LLY, AVGO, MU, NFLX, COST, GE, TSLA) from 2025-05-01, every seventh session as an entry at the open, with the stop 3 x the
simple 14-session ATR under the previous close (the trend family's own distance), no target, a 63-session time stop, followed twice (a fixed stop and a trailing one): the first start whose trailing stop closed more than 10 points above where the fixed stop
closed is MICROSOFT, entered at the open of 2026-03-26 (370.815): a run-up to 432.92 on 2026-04-22 and then a drop. The ATR of the day before was 7.1603, so the distance 21.4810 and the starting stop 349.5590 (the previous close 371.04 less that). The fixed stop only
closed on 2026-06-25 at 349.5590 (-5.7%); the trailing stop closed on 2026-04-23 at 411.4390 (+11.0%). The distances are the plan's own parameters; the bars are the market's.
"""
from datetime import date
from decimal import Decimal

import pytest
from proving_helpers import real_bars, stored_bars_available

from prometheus.agents.trader import simulator
from prometheus.agents.trader.record import Fill, Trade
from prometheus.common.money import price as price_door, price_scale
from prometheus.common.portfolio import trade_amount
from prometheus.proving import live, outcomes, record
from prometheus.proving.portfolio import LotInfo, SimPortfolio

pytestmark = pytest.mark.skipif(not stored_bars_available(), reason="the stored daily bars (data/historical/day_aggs_v1) are not on this machine")

ENTRY = "2026-03-26"
STOP, TRAIL = Decimal("349.5590"), Decimal("21.4810")
TIME_STOP = "2026-06-26"
LEVELS = {      # the stop in force on each bar, by hand: the larger of 349.5590 and (the highest close of the bars before it) - 21.4810 (the table of the bars of 2026-03-26 to 2026-04-23)
    "2026-03-26": "349.5590", "2026-04-06": "351.9790", "2026-04-09": "352.8490", "2026-04-14": "362.8890", "2026-04-15": "371.6290", "2026-04-16": "389.7390", "2026-04-17": "398.7790", "2026-04-20": "401.3090",
    "2026-04-22": "402.6790", "2026-04-23": "411.4390"}


@pytest.fixture(scope="module")
def msft():
    bars = real_bars(["MSFT", "SPY"])
    i = next(k for k, b in enumerate(bars["MSFT"]) if b["date"] == ENTRY)
    return bars["MSFT"][i:i + 70], bars["SPY"]


def by_hand(series, stop, trail):
    """The trailing stop worked out here, with no `simulator` call but the price door: (the index of the bar that stops it, its date, the price it is sold at, the stop in force on each bar up to it). A gap through the stop is sold at the open."""
    highest, levels = None, []
    for index, bar in enumerate(series):
        level = stop if highest is None else max(stop, price_scale(highest - trail))
        levels.append(level)
        opened, low, close = (price_door(bar[k]) for k in ("open", "low", "close"))
        if opened <= level:
            return index, bar["date"], opened, levels
        if low <= level:
            return index, bar["date"], level, levels
        highest = close if highest is None else max(highest, close)
    return None, None, None, levels


def test_a_trailing_stop_rises_with_the_highest_close_never_falls_and_closes_the_trade_above_the_plans_own_stop(msft):
    series, spy = msft
    index, day, sold, levels = by_hand(series, STOP, TRAIL)
    assert (index, day, sold) == (19, "2026-04-23", Decimal("411.4390")) and levels == sorted(levels) and levels[0] == STOP and levels[-1] > STOP      # the stop only ever rises, and starts at the plan's stop
    assert {series[i]["date"]: str(level) for i, level in enumerate(levels) if series[i]["date"] in LEVELS} == LEVELS
    entry = price_door(series[0]["open"])
    out = simulator.follow(entry, Decimal(10), ENTRY, series, STOP, None, TIME_STOP, Decimal(0), spy, trail_distance=TRAIL)
    assert (out.status, out.exit_reason, out.exit_date, out.exit_price, out.sessions_held) == ("closed", "trailing_stop", day, sold, index)
    assert out.exit_price > entry and out.realized_gain == trade_amount(Decimal(10), sold) - trade_amount(Decimal(10), entry) == Decimal("406.24") and out.return_pct == 0.109553
    assert out.benchmark_return_pct == simulator._benchmark(spy, ENTRY, day) and out.excess_return_pct == simulator._return_difference(out.return_pct, out.benchmark_return_pct)      # the benchmark is read as for any plan
    assert isinstance(out.exit_price, Decimal) and out.exit_price == price_scale(out.exit_price)                       # a price at four places, a Decimal


def test_the_fixed_stop_of_a_plan_with_no_trail_is_followed_exactly_as_before(msft):
    """E8: no `trail_distance` is the old follower. The same bars and the same plan with a fixed stop close on 2026-06-25 at the stop (the exit reason `stop`, never `trailing_stop`)."""
    series, _ = msft
    entry = price_door(series[0]["open"])
    fixed = simulator.follow(entry, Decimal(10), ENTRY, series, STOP, None, TIME_STOP, Decimal(0))
    assert (fixed.exit_reason, fixed.exit_date, fixed.exit_price, fixed.return_pct, fixed.sessions_held) == ("stop", "2026-06-25", STOP, -0.057322, 62)
    assert simulator.follow(entry, Decimal(10), ENTRY, series, STOP, None, TIME_STOP, Decimal(0), None, trail_distance=None) == fixed


def test_a_days_own_close_never_moves_its_own_stop_so_nothing_is_used_from_the_day_being_judged(msft):
    """The stop of the bar of 2026-04-23 is made from the closes up to 2026-04-22 (432.92). Cut at 2026-04-22 the position is still open on that day, and the stop that closes the trade on the 23rd is the one the 22nd's close made (411.4390), not one
    made from the 23rd's own close (415.75 would give 394.2690)."""
    series, _ = msft
    cut = [b for b in series if b["date"] <= "2026-04-22"]
    entry = price_door(series[0]["open"])
    held = simulator.follow(entry, Decimal(10), ENTRY, cut, STOP, None, TIME_STOP, Decimal(0), trail_distance=TRAIL)
    assert held.status == "open" and held.last_date == "2026-04-22" and held.sessions_held == 18
    through = simulator.follow(entry, Decimal(10), ENTRY, series, STOP, None, TIME_STOP, Decimal(0), trail_distance=TRAIL)
    assert through.exit_price == Decimal("411.4390") and price_scale(price_door(series[19]["close"]) - TRAIL) == Decimal("394.2690") < through.exit_price


def test_the_trailing_stop_is_the_one_place_the_stop_in_force_is_worked_out_and_a_float_is_refused():
    level = simulator.trailing_stop
    assert level(Decimal("100"), None, Decimal("10")) == Decimal("100")                           # no close yet: the plan's stop
    assert level(Decimal("100"), Decimal("105"), Decimal("10")) == Decimal("100")                  # the highest close less the distance (95) is under the plan's stop: the stop does not fall
    assert level(Decimal("100"), Decimal("130.1234"), Decimal("10.5")) == Decimal("119.6234")     # above it: the highest close less the distance, at four places
    assert level(None, Decimal("130"), Decimal("10")) == Decimal("120") and level(None, None, Decimal("10")) is None
    assert level(Decimal("100"), Decimal("130"), None) == Decimal("100") and level(None, Decimal("130"), None) is None      # a plan with a fixed stop has no trail
    with pytest.raises(TypeError):
        level(100.0, Decimal("130"), Decimal("10"))                                                # a float is never a price here (entry 112): it crosses a door first


def test_a_trailing_plan_is_read_from_a_record_and_an_old_record_has_a_fixed_stop():
    trend = {"symbol": "MSFT", "trade": {"family": "trend", "entry_mid": 371.04, "stop_price": 349.559, "stop_distance": 21.481, "target_price": None, "trailing": True, "trail_distance": 21.481, "time_stop_date": TIME_STOP, "horizon_bars": 63}}
    plan = record.plan_of(trend)
    assert (plan.trailing, plan.trail_distance, plan.stop, plan.target) == (True, Decimal("21.4810"), Decimal("349.5590"), None)
    old = record.plan_of({"symbol": "MSFT", "trade": {k: v for k, v in trend["trade"].items() if k not in ("trailing", "trail_distance")}})
    assert (old.trailing, old.trail_distance, old.stop) == (False, None, Decimal("349.5590"))      # a record made before Advisor config version 3
    assert record.plan_of({"symbol": "MSFT", "trade": {**trend["trade"], "trail_distance": None}}).trailing is False      # a plan that says it trails but has no distance cannot
    assert record.plan_of({"symbol": "MSFT", "trade": {**trend["trade"], "trailing": False}}).trail_distance is None


def real_plan() -> record.PlanView:
    return record.PlanView(entry_low=Decimal("367.4600"), entry_high=Decimal("374.6200"), entry_mid=Decimal("371.0400"), stop=STOP, trailing=True, trail_distance=TRAIL, time_stop_date=date.fromisoformat(TIME_STOP), horizon_bars=63)


def test_an_outcome_of_a_trailing_plan_is_won_by_a_trailing_stop_and_its_path_is_checked_against_the_follower(msft):
    """`outcomes.score` on the real bars, decision day 2026-03-25 (MSFT's close 371.04): entered at the next open, closed by the trailing stop on 2026-04-23 at 411.4390 for a return of +10.96%; the day-by-day path is checked against `follow` inside `score`."""
    bars = real_bars(["MSFT", "SPY"])
    out = outcomes.score(real_plan(), bars["MSFT"], bars["SPY"], None, date(2026, 3, 25), [5, 10, 20], symbol="MSFT")
    assert (out.result, out.hypothetical, out.entry.date.isoformat(), out.entry.price) == ("won", True, ENTRY, Decimal("370.8150"))
    assert (out.exit.reason, out.exit.date.isoformat(), out.exit.price, out.exit.sessions_held, out.return_pct) == ("trailing_stop", "2026-04-23", Decimal("411.4390"), 19, 0.109553) and out.plan.trailing is True
    assert len(out.path) == 20 and out.path[-1].date.isoformat() == "2026-04-23" and out.max_favorable_pct >= 0.16     # the best close 432.92 was 16.7% over the entry
    fixed = outcomes.score(real_plan().model_copy(update={"trailing": False, "trail_distance": None}), bars["MSFT"], bars["SPY"], None, date(2026, 3, 25), [5, 10, 20], symbol="MSFT")
    assert (fixed.exit.reason, fixed.exit.date.isoformat(), fixed.return_pct, fixed.result) == ("stop", "2026-06-25", -0.057322, "lost")      # the same recommendation by a fixed stop: the trailing one kept the gain


def trade_of(price: Decimal) -> Trade:
    fill = Fill(price=price, shares=Decimal(10), amount=trade_amount(Decimal(10), price), date=ENTRY, price_source="bar_open")
    return Trade(trade_id="T1", kind="recommendation", record_id="r1", symbol="MSFT", side="buy", shares_approved=Decimal(10), shares=Decimal(10), status="filled", mode="backtest", fill=fill)


def test_the_time_machines_account_follows_a_trailing_lot_by_the_same_follower_and_shows_the_stop_in_force(msft):
    """`SimPortfolio` walked day by day over the real bars: after each close the position's `stop` is the stop that will be in force on the NEXT bar (the plan's stop or the highest close so far less 21.4810) and it is marked `trailing`; on 2026-04-23 it is sold at 411.4390
    for `trailing_stop`, the gain being the follower's own."""
    series, spy = msft
    bars = {"MSFT": series, "SPY": spy}
    entry = price_door(series[0]["open"])
    account = SimPortfolio(Decimal("5000.00"))
    account.apply_fills([trade_of(entry)], {"r1": LotInfo(plan=real_plan(), key="k", decision_day=date(2026, 3, 25))})
    highest, shown, sold = None, [], []
    for bar in series[:20]:
        day = date.fromisoformat(bar["date"])
        sold += account.step(day, bars)
        account.mark(day, bars)
        close = price_door(bar["close"])
        highest = close if highest is None else max(highest, close)
        if not sold:
            (position,) = account.positions(day)
            assert position.trailing is True and position.target is None and position.stop == max(STOP, price_scale(highest - TRAIL)), bar["date"]
            shown.append(position.stop)
    assert shown == sorted(shown) and shown[0] == STOP and shown[-1] == Decimal("411.4390") and len(shown) == 19      # the last stop shown (after the close of the 22nd) is the one that closes the trade on the 23rd
    (exit_,) = sold
    assert (exit_.reason, exit_.date.isoformat(), exit_.price, exit_.sessions_held, exit_.amount, exit_.gain) == ("trailing_stop", "2026-04-23", Decimal("411.4390"), 19, Decimal("4114.39"), Decimal("406.24"))
    assert account.lots == [] and account.cash == Decimal("5000.00") + Decimal("406.24")


def test_a_trailing_lot_that_crosses_a_real_split_keeps_its_stop_in_the_share_basis_of_today():
    """Netflix split 10 for 1 on 2025-11-17 (the table is the one of the portfolio tests). A purchase of 3 shares at the real open of 2025-11-14 with a trailing plan made of the real stop distances: the lot is followed in the basis of its entry, so the split is
    no stop, and after the split the stop shown is the entry-basis stop in force divided by ten (a price at four places)."""
    from proving_helpers import close_on, real_records, restated

    from prometheus.proving import splits
    bars = real_bars({"NFLX"})
    net = bars["NFLX"]
    plan = restated(record.plan_of(next(r for r in real_records() if r.record_id == "83910656bece").rec), close_on(net, "2025-11-13"), date(2025, 11, 13))
    plan = plan.model_copy(update={"target": None, "trailing": True, "trail_distance": plan.entry_mid - plan.stop})
    opened = simulator.bar_price(simulator.bar_on(net, "2025-11-14"), "open")
    fill = Fill(price=opened, shares=Decimal(3), amount=trade_amount(Decimal(3), opened), date="2025-11-14", price_source="bar_open")
    trade = Trade(trade_id="T1", kind="recommendation", record_id="r1", symbol="NFLX", side="buy", shares_approved=Decimal(3), shares=Decimal(3), status="filled", mode="backtest", fill=fill)
    table = splits.SplitTable(symbol="NFLX", splits=[splits.Split(execution_date=date(2025, 11, 17), split_from=1.0, split_to=10.0)])
    account = SimPortfolio(Decimal("10000.00"), splits={"NFLX": table})
    account.apply_fills([trade], {"r1": LotInfo(plan=plan, key="k", decision_day=date(2025, 11, 13))})
    assert account.lots[0].trail == plan.trail_distance and account.lots[0].stop == plan.stop
    highest, after = None, None
    for day in [date.fromisoformat(b["date"]) for b in bars["SPY"] if "2025-11-14" <= b["date"] <= "2025-11-20"]:
        account.step(day, bars)
        account.mark(day, bars)
        bar = simulator.bar_on(net, day.isoformat())
        close = simulator.bar_price(bar, "close") * (10 if day >= date(2025, 11, 17) else 1)        # the closes in the share basis of the entry
        highest = close if highest is None else max(highest, close)
        assert account.lots, f"the position was sold on {day}: the split read as a stop"
        if day == date(2025, 11, 17):
            level = max(plan.stop, price_scale(highest - plan.trail_distance))
            after = account.positions(day)[0]
            assert after.stop == price_scale(level / 10) and after.shares == Decimal(30) and after.trailing is True
    assert after is not None and account.closed == []


def test_the_live_view_watches_a_trailing_stop_against_the_stop_in_force_on_each_bar_and_reports_its_level(msft):
    """P11 and E3: `live._judge` on the real bars of the position bought on 2026-03-26. Judged on 2026-04-22 it is in its plan; on 2026-04-23 the low of 411.4101 is under the stop the 22nd's close made (411.4390): STOP HIT, with that level, in the sentence's words,
    and nothing is sold. The same bars judged by a FIXED stop of 349.5590 are in plan on the 23rd. `_trail_note` says where the stop stands."""
    series, _ = msft
    since, suspect = date.fromisoformat(ENTRY), Decimal("0.40")
    plan = live.LivePlan(stop=STOP, trailing=True, trail_distance=TRAIL, time_stop_date=date.fromisoformat(TIME_STOP))
    on_22 = [b for b in series if b["date"] <= "2026-04-22"]
    status, evidence, _, level = live._judge("MSFT", plan, price_door(on_22[-1]["close"]), on_22, since, date(2026, 4, 22), suspect)
    assert (status, evidence, level) == ("in_plan", None, None)
    on_23 = [b for b in series if b["date"] <= "2026-04-23"]
    status, evidence, _, level = live._judge("MSFT", plan, price_door(on_23[-1]["close"]), on_23, since, date(2026, 4, 23), suspect)
    assert (status, evidence, level) == ("stop_hit", "the low of 2026-04-23", Decimal("411.4390"))
    position = live.LivePosition(symbol="MSFT", shares=Decimal(10), cost_basis=Decimal("370.8150"), price=price_door(on_23[-1]["close"]), plan=plan.model_copy(update={"stop_now": level, "highest_close": Decimal("432.9200")}), status=status, evidence=evidence)
    assert live._sentence(position, level) == "MSFT hit its trailing stop at 411.4390 (the low of 2026-04-23): the price now is 415.7500. Nothing was sold; selling is your decision."
    fixed = live.LivePlan(stop=STOP, time_stop_date=date.fromisoformat(TIME_STOP))
    assert live._judge("MSFT", fixed, price_door(on_23[-1]["close"]), on_23, since, date(2026, 4, 23), suspect)[0] == "in_plan"
    raised = plan.model_copy(update={"highest_close": Decimal("432.9200"), "stop_now": Decimal("411.4390")})
    assert live._trail_note(raised) == "the trailing stop is now 411.4390 (highest close 432.9200 - 21.4810)"
    assert live._trail_note(plan.model_copy(update={"highest_close": Decimal("360.0000"), "stop_now": STOP})) == "the trailing stop is still the plan's own, 349.5590: no close has lifted it yet"
    assert live._trail_base(on_23, since, "2026-04-23") == Decimal("432.9200") and live._trail_base(on_23, None, "2026-04-23") is None


def test_the_critics_reward_to_risk_reads_no_target_so_a_trailing_plan_is_scored_as_any_other_and_never_as_zero_or_unknown():
    """E7. The Critic's `reward_to_risk` is the distance up to the 126-session high over the stop distance: it reads no `target_price`. The same real recommendation (PG, `PG_open`) with a target and as a TRAILING plan (no target, `trailing`, a trail distance) gets the same
    measure, the same score and the same detail: available, never counted as unknown credit, never zero."""
    from critic_helpers import CONFIG, captured, rec_of, tamper

    from prometheus.agents.critic import value as vl
    fetched = {k: {"value": v} for k, v in dict(sma_126=100.0, high_126d=105.0, low_126d=90.0, earnings_yield=0.05, eps=5.0, ps_ratio=4.0, ret_21d=0.01, max_drawdown_63d=-0.05).items()}

    def judged(**changes):
        order = captured("PG_open")
        order.recommendations[0] = tamper(order.recommendations[0], **{"trade.stop_distance": 5.0, **changes})
        return vl.assess(rec_of(order), fetched, 88.0, CONFIG.value, CONFIG.verdict.unknown_credit)
    fixed = judged(**{"trade.target_price": 110.0})
    trailing = judged(**{"trade.target_price": None, "trade.trailing": True, "trade.trail_distance": 5.0})
    assert [m.model_dump() for m in trailing.measures] == [m.model_dump() for m in fixed.measures] and trailing.score == fixed.score and trailing.detail == fixed.detail and trailing.applicable is True
    reward = next(m for m in trailing.measures if m.name == "reward_to_risk")
    assert reward.available is True and reward.x == 3.4 and reward.value > 0 and reward.note is None      # (105 - 88) / 5, a real measure
