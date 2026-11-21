"""`outcomes.score`, `splits` and the readers of a recommendation (design/proving_ground_spec.md sections 5, 8 and 14; design choices entry 126; P4, P5, P8, P10).

REAL DATA ONLY. The recommendations are the 28 distinct records of the real captures (`tests/fixtures/session_real/`) and of the real ledger (`data/ledger/recommendations.jsonl`), made on 2026-11-06, 2026-11-12 and 2026-11-13, and the bars are the stored daily
bars (`data/historical/day_aggs_v1`, read only) to the finished session of 2026-11-19 (`proving_helpers.real_bars`). A machine without those bars skips these tests. Nothing here calls a model or a paid API.

THE FEW DAYS SINCE NOVEMBER DO NOT EXERCISE EVERY RULE, so where a case needs a longer time (a time stop, a limit the day never reached, a gap through a stop or a target, a split) the test takes a REAL plan of those records and puts its OWN
distances (its stop, target, entry zone and limit price as fractions of its entry price, and its horizon in sessions, by the Advisor's own `add_weekdays`) on an EARLIER REAL entry day of the same symbol in the stored bars: `restated()`. Every
price that is then compared with is a real stored bar; only the plan's distances are carried back. The splits are real too (Netflix 10 for 1 on 2025-11-17 is in the stored bars: a close of 1112.17 and an open of 110.75); the table is written in the
shape `splits.py` documents, from those public facts, and the test checks it against the bars.

Each exit is read THREE ways: by `outcomes.score`, by `simulator.follow` called by hand on the same bars, and by a ten-line reading of the plan's rules written in this file (the stop before the target, only the stop on the entry bar, the time stop
at the first bar on or after its date).
"""
import json
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest
from proving_helpers import FIRST, LAST, close_on, real_bars, real_records, restated, stored_bars_available

from prometheus.agents.trader import simulator
from prometheus.agents.trader.config import load_config as trader_config
from prometheus.common.money import price as price_door
from prometheus.proving import outcomes, record, splits
from prometheus.proving.config import load_config
from prometheus.proving.outcomes import EntryFill, entry_fill_of

pytestmark = pytest.mark.skipif(not stored_bars_available(), reason="the stored daily bars (data/historical/day_aggs_v1) are not on this machine")

HORIZONS = [5, 10, 20]
QUARTER = Decimal("0.0001")
RETURN = Decimal("0.000001")


def d(x) -> Decimal:
    return Decimal(str(x))


def four(x: Decimal) -> Decimal:
    return x.quantize(QUARTER, rounding=ROUND_HALF_UP)


def bars_for(reals) -> dict:
    return real_bars({r.rec["symbol"] for r in reals} | {r.rec["identity"].get("sector_etf") for r in reals if r.rec["identity"].get("sector_etf")})


def hand_entry(plan, series, day: date):
    """The entry as the Trader's ticket and the stored bar make it, worked out by hand from the documented rules: the first bar after the decision day; the limit range is the entry zone cut to 1% above and 3% below the plan's own price;
    the order is a limit at the plan's limit price inside it; it fills at the open when the open is not above it, at the limit when the day traded down to it, else not at all. Returns (bar, price or None, reason)."""
    config = trader_config().orders
    bar = next(b for b in series if b["date"] > day.isoformat())
    low, high = max(plan.entry_low, four(plan.entry_mid * (1 - config.limit_band_pct))), min(plan.entry_high, four(plan.entry_mid * (1 + config.max_chase_pct)))
    limit = four(min(max(plan.limit_price, low), high))
    if d(bar["open"]) <= limit:
        return bar, four(d(bar["open"])), "open"
    if d(bar["low"]) <= limit:
        return bar, limit, "limit"
    return bar, None, "limit_not_reached"


def hand_exit(series, entry_date: str, stop, target, time_stop: str | None):
    """The first rule that fires, by hand: (sessions held, reason, price, date) or None. A gap through the stop sells at the open; the target and the time stop cannot fire on the entry bar; a gap through the target sells at the open."""
    for index, bar in enumerate([b for b in series if b["date"] >= entry_date]):
        o, h, low, c = (d(bar[k]) for k in ("open", "high", "low", "close"))
        if stop is not None and o <= stop:
            return index, "stop", four(o), bar["date"]
        if stop is not None and low <= stop:
            return index, "stop", stop, bar["date"]
        if index > 0 and target is not None and o >= target:
            return index, "target", four(o), bar["date"]
        if index > 0 and target is not None and h >= target:
            return index, "target", target, bar["date"]
        if index > 0 and time_stop is not None and bar["date"] >= time_stop:
            return index, "time_stop", four(c), bar["date"]
    return None


def score(real_plan, bars, symbol, day: date, sector=None, **more):
    return outcomes.score(real_plan, bars[symbol], bars["SPY"], bars.get(sector) if sector else None, day, HORIZONS, symbol=symbol, **more)


def test_every_real_recommendation_is_scored_and_each_entry_and_exit_agrees_with_follow_and_with_a_reading_of_the_plan_by_hand():
    reals = real_records()
    bars = bars_for(reals)
    assert len(reals) == 28 and {r.rec["identity"]["schema_version"] for r in reals} == {2, 3}
    results = []
    for real in reals:
        symbol, plan = real.rec["symbol"], record.plan_of(real.rec)
        out = score(plan, bars, symbol, real.day, real.rec["identity"].get("sector_etf"), identity=record.identity_of(real.order, real.rec, real.day))
        series = bars[symbol]
        bar, price, how = hand_entry(plan, series, real.day)
        assert out.hypothetical and not out.traded and out.gain is None and out.entry.shares == 1 and out.record_id == real.record_id and out.decision_day == real.day
        assert (out.entry.date.isoformat(), out.entry.price) == (bar["date"], price), (real.record_id, how)
        by_hand = hand_exit(series, bar["date"], plan.stop, plan.target, plan.time_stop_date.isoformat())
        follow = simulator.follow(price, Decimal(1), bar["date"], series, plan.stop, plan.target, plan.time_stop_date.isoformat(), Decimal(0), bars["SPY"])
        if by_hand is None:
            assert out.result == "open" and out.exit is None and follow.status == "open"
            assert out.path[-1].date.isoformat() == follow.last_date == series[-1]["date"] and len(out.path) - 1 == follow.sessions_held
            assert out.return_pct == float(((d(series[-1]["close"]) / price) - 1).quantize(RETURN, rounding=ROUND_HALF_UP))
        else:
            index, reason, sold, when = by_hand
            assert (out.exit.sessions_held, out.exit.reason, out.exit.price, out.exit.date.isoformat()) == (index, reason, sold, when)
            assert (follow.sessions_held, follow.exit_reason, follow.exit_price, follow.exit_date) == (index, reason, sold, when)
            assert out.result == ("won" if sold > price else "lost") and out.return_pct == float(((sold / price) - 1).quantize(RETURN, rounding=ROUND_HALF_UP)) and out.return_pct == follow.return_pct
            assert [p.date.isoformat() for p in out.path][-1] == when and len(out.path) == index + 1
        assert out.benchmark_return_pct == follow.benchmark_return_pct and out.excess_return_pct == follow.excess_return_pct          # the benchmark's return is the simulator's own, from the open of the entry bar
        assert (out.verdict == "unreviewed") == (out.critic is None) and out.advisor.score == real.rec["score"] and out.advisor.family == real.rec["trade"]["family"]       # the ledger holds the Critic's review of only three of its seven
        results.append(out.result)
    assert results.count("open") == 6 and results.count("won") == 5 and results.count("lost") == 17        # 22 closed, 6 open: the closed ones are a sample over the minimum of 20 (P9)


PINS = [      # record_id: (symbol, decision day, result, entry date, entry price, exit date, exit price, reason, sessions, return, benchmark, excess, sector, horizons (5 sessions), max favorable, max adverse): from the stored bars, read on 2026-11-20
    ("83910656bece", "AMD", "2026-11-13", "won", "2026-11-16", "503.4450", "2026-11-19", "542.1700", "target", 3, 0.07692, -0.045074, 0.121994, -0.036284, (0.07692, -0.045074), 0.07692, -0.028315),
    ("c17c56d1ce7d", "CVX", "2026-11-13", "lost", "2026-11-16", "220.0300", "2026-11-17", "215.4814", "stop", 1, -0.020673, -0.02297, 0.002297, -0.017979, (-0.020673, -0.02297), -0.013135, -0.020673),
    ("8b78a7afdbcc", "XOM", "2026-11-13", "lost", "2026-11-16", "169.5300", "2026-11-16", "169.3786", "stop", 0, -0.000893, -0.012953, 0.01206, -0.012864, (-0.000893, -0.012953), -0.000893, -0.01569),
    ("d415bba65829", "EOG", "2026-11-13", "lost", "2026-11-16", "156.5100", "2026-11-18", "150.0280", "stop", 2, -0.041416, -0.025958, -0.015458, -0.050062, (-0.041416, -0.025958), -0.006645, -0.043575),
    ("82c3c64ceb62", "LLY", "2026-11-12", "won", "2026-11-13", "1429.9100", "2026-11-16", "1473.5183", "target", 1, 0.030497, -0.026471, 0.056968, None, (0.030497, -0.026471), 0.030497, 0.001231),
    ("a497639f3127", "QSR", "2026-11-06", "lost", "2026-11-09", "78.0300", "2026-11-16", "74.7294", "stop", 5, -0.042299, -0.002696, -0.039603, -0.012867, (-0.042299, -0.002696), -0.005126, -0.042299),
    ("02361b7f0ef9", "MCD", "2026-11-06", "won", "2026-11-09", "276.9800", "2026-11-10", "283.1355", "target", 1, 0.022224, 0.010742, 0.011482, 0.011363, (0.022224, 0.010742), 0.022224, 0.016102),
    ("be3e391cac5f", "MSFT", "2026-11-13", "open", "2026-11-16", "523.8800", None, None, None, None, -0.048828, -0.045074, -0.003754, -0.036284, (None, None), -0.011243, -0.048828),
    ("1b7f1c582e26", "MCD", "2026-11-06", "open", "2026-11-09", "276.9800", None, None, None, None, -0.016716, -0.035151, 0.018435, -0.031832, (-0.00834, -0.002696), 0.018738, -0.018088),
]


@pytest.mark.parametrize("pin", PINS, ids=lambda p: f"{p[1]}-{p[2]}-{p[3]}")
def test_the_real_plans_of_november_end_as_the_stored_bars_say_a_stop_a_target_a_stop_on_the_entry_bar_and_a_position_still_open(pin):
    """The numbers the page shows for these real recommendations, from the stored bars: the entry at the next open (a limit order that the open is below fills at the open), the exit the plan's own, the return, the benchmark's from the open of
    the entry bar to the close of the exit day (the open ones: to the last close), the sector fund's, the 5-session mark (a closed position keeps its realized return, an open one is marked at that session's close, and None while it has not been held that
    long), the best and the worst close."""
    rid, symbol, day, result, entry_day, entry_price, exit_day, exit_price, reason, sessions, ret, bench, excess, sector, five, best, worst = pin
    real = next(r for r in real_records() if r.record_id == rid)
    bars = bars_for([real])
    out = score(record.plan_of(real.rec), bars, symbol, date.fromisoformat(day), real.rec["identity"].get("sector_etf"))
    assert (out.result, out.entry.date.isoformat(), str(out.entry.price)) == (result, entry_day, entry_price)
    assert (out.return_pct, out.benchmark_return_pct, out.excess_return_pct, out.sector_return_pct, out.max_favorable_pct, out.max_adverse_pct) == (ret, bench, excess, sector, best, worst)
    assert out.horizons[0].sessions == 5 and (out.horizons[0].return_pct, out.horizons[0].benchmark_return_pct) == five and [h.sessions for h in out.horizons] == HORIZONS
    if reason is None:
        assert out.exit is None and out.path[-1].date.isoformat() == "2026-11-19" and out.path[-1].return_pct == ret and out.path[-1].benchmark_return_pct == bench          # marked at the last stored close
    else:
        assert (out.exit.date.isoformat(), str(out.exit.price), out.exit.reason, out.exit.sessions_held) == (exit_day, exit_price, reason, sessions) and [h.return_pct for h in out.horizons] == [ret] * 3
    assert out.path[0].date.isoformat() == entry_day and out.path[0].return_pct == float((d(next(b for b in bars[symbol] if b["date"] == entry_day)["close"]) / out.entry.price - 1).quantize(RETURN, rounding=ROUND_HALF_UP))


def test_a_real_plans_own_distances_on_an_earlier_real_day_reach_the_time_stop_and_a_limit_the_day_never_reached_and_a_gap_through_a_stop_and_a_target():
    """Each rule that the days since November do not reach, at a real earlier day of the same symbol (see the top of this file): a time stop that sells at the close of its day (MCD, a loss; SNDK, a win of 4.2 times), a limit that the next day never
    traded down to (Exxon's open of 105.34 and low of 105.34 against a limit of 105.2706: not filled, with the Trader's own code), an order filled AT ITS LIMIT because the day opened above it and traded down to it, a gap through the stop (sold at the OPEN, 110.36,
    below the stop) and a gap through the target (sold at the open, 110.27, above the target)."""
    reals = {r.record_id: r for r in real_records()}
    bars = real_bars({"MCD", "SNDK", "XOM", "XLE", "XLY"})
    for rid, symbol, day, entry, exit_on, exit_price, ret, bench, sessions in [("1b7f1c582e26", "MCD", "2025-10-17", "2025-10-20", "2026-04-13", "304.5100", -0.011171, 0.028142, 119),
                                                                                ("816aea79fcda", "SNDK", "2025-05-02", "2025-05-05", "2025-10-27", "176.4900", 4.216967, 0.218053, 121)]:
        plan = restated(record.plan_of(reals[rid].rec), close_on(bars[symbol], day), date.fromisoformat(day))
        out = score(plan, bars, symbol, date.fromisoformat(day))
        assert (out.entry.date.isoformat(), out.exit.date.isoformat(), str(out.exit.price), out.exit.reason, out.exit.sessions_held, out.return_pct, out.benchmark_return_pct) == (entry, exit_on, exit_price, "time_stop", sessions, ret, bench)
        assert hand_exit(bars[symbol], entry, plan.stop, plan.target, plan.time_stop_date.isoformat())[1:] == ("time_stop", Decimal(exit_price), exit_on)
        assert out.exit.date >= plan.time_stop_date and min(d(b["low"]) for b in bars[symbol] if entry <= b["date"] <= exit_on) > plan.stop       # the stop was never touched on the way
        assert out.result == ("lost" if ret < 0 else "won") and len(out.path) == sessions + 1 and [h.sessions for h in out.horizons] == HORIZONS
    xom = record.plan_of(reals["8b78a7afdbcc"].rec)
    cases = {}
    for day in ("2025-05-07", "2025-05-01", "2025-07-28", "2025-05-08"):
        plan = restated(xom, close_on(bars["XOM"], day), date.fromisoformat(day))
        cases[day] = (plan, score(plan, bars, "XOM", date.fromisoformat(day), "XLE"))
    plan, out = cases["2025-05-07"]
    bar = next(b for b in bars["XOM"] if b["date"] == "2025-05-08")
    assert (out.result, out.reject_code, out.entry, out.exit, out.path, out.return_pct, out.gain) == ("not_filled", "limit_not_reached", None, None, [], None, None)
    assert d(bar["open"]) > plan.limit_price and d(bar["low"]) > plan.limit_price and out.notes == ["the limit 105.2706 was not reached on 2025-05-08 (a day order): the day's low was 105.3400"]
    plan, out = cases["2025-05-01"]
    bar = next(b for b in bars["XOM"] if b["date"] == "2025-05-02")
    assert d(bar["open"]) > plan.limit_price >= d(bar["low"]) and (str(out.entry.price), out.entry.date.isoformat()) == ("106.4480", "2025-05-02")       # filled at its limit, not at the open of 107.72
    assert (out.result, out.exit.reason, out.exit.sessions_held, str(out.exit.price), out.return_pct) == ("lost", "stop", 0, "105.0288", -0.013332)      # and stopped on the entry bar itself, by the stop only
    plan, out = cases["2025-07-28"]
    assert (out.exit.date.isoformat(), str(out.exit.price), out.exit.reason) == ("2025-07-31", "110.3600", "stop") and d(next(b for b in bars["XOM"] if b["date"] == "2025-07-31")["open"]) <= plan.stop
    assert out.exit.price < plan.stop                                                                      # a gap through the stop sells at the open, below it
    plan, out = cases["2025-05-08"]
    assert (out.exit.date.isoformat(), str(out.exit.price), out.exit.reason, out.result) == ("2025-05-12", "110.2700", "target", "won") and d(next(b for b in bars["XOM"] if b["date"] == "2025-05-12")["open"]) >= plan.target
    assert out.exit.price > plan.target and out.sector_return_pct == 0.023971 and out.excess_return_pct == 0.003928                 # a gap through the target sells at the open, above it


def test_a_real_fill_is_the_entry_as_it_was_and_the_gain_is_money_where_a_hypothetical_outcome_has_none():
    """The desk really bought these (the captures' paper fills at the quote of 2026-11-13, shares and price as the Trader wrote them): the outcome is traded, its entry is the fill, its gain is `trade_amount` to the cent (Decimal), and the position is
    followed from the bar of the fill's day, where only the stop can fire. Exxon's plan stop of 169.3786 lies above that day's low, so it was stopped the day it was bought: 8 shares at 170.6753 (1,365.40) sold at 169.3786 (1,355.03): -10.37."""
    reals = {r.record_id: r for r in real_records()}
    bars = real_bars({"XOM", "EOG", "MSFT", "XLE", "XLK"})
    xom = reals["8b78a7afdbcc"]
    assert xom.fill == EntryFill(date(2026, 11, 13), Decimal("170.6753"), Decimal(8))
    out = score(record.plan_of(xom.rec), bars, "XOM", xom.day, "XLE", fill=xom.fill, identity=record.identity_of(xom.order, xom.rec, xom.day))
    assert out.traded and not out.hypothetical and out.not_traded_reason is None and out.verdict == "approved"
    assert (out.entry.date.isoformat(), out.entry.price, out.entry.shares, out.entry.amount) == ("2026-11-13", Decimal("170.6753"), Decimal(8), Decimal("1365.40"))
    assert (out.exit.date.isoformat(), out.exit.price, out.exit.reason, out.exit.sessions_held, out.result) == ("2026-11-13", Decimal("169.3786"), "stop", 0, "lost")
    assert out.gain == Decimal("-10.37") == simulator.trade_amount(Decimal(8), Decimal("169.3786")) - Decimal("1365.40") and isinstance(out.gain, Decimal) and out.return_pct == -0.007597
    msft = reals["be3e391cac5f"]
    still = score(record.plan_of(msft.rec), bars, "MSFT", msft.day, "XLK", fill=msft.fill)
    assert still.result == "open" and still.exit is None and still.entry.amount == Decimal("7858.20") and still.gain == Decimal("-383.70")      # the unrealized gain at the last stored close, 498.30: 15 x (498.30 - 523.88)
    hypothetical = score(record.plan_of(msft.rec), bars, "MSFT", msft.day, "XLK")
    assert hypothetical.hypothetical and hypothetical.gain is None and hypothetical.entry.shares == 1 and hypothetical.entry.date.isoformat() == "2026-11-16"
    assert entry_fill_of({**{"trade_id": "T1", "kind": "recommendation", "symbol": "XOM", "side": "buy", "shares_approved": 8, "shares": 8, "status": "rejected", "reject_code": "insufficient_cash", "mode": "paper"}}) is None


def test_no_bar_no_outcome_and_a_price_is_never_invented():
    """A symbol with no stored bar (a London listing) and a decision on the last stored day (the future is not stored) are `no_data`, each with its reason, no entry, no exit and no number; and a decision whose next session is the last stored one
    is `open` with the one day it has."""
    reals = {r.record_id: r for r in real_records()}
    plan = record.plan_of(reals["8b78a7afdbcc"].rec)
    bars = real_bars({"XOM", "MSFT", "SHEL.L"})
    assert bars["SHEL.L"] == []
    nothing = outcomes.score(plan, bars["SHEL.L"], bars["SPY"], None, date(2026, 11, 13), HORIZONS, symbol="SHEL.L")
    assert (nothing.result, nothing.entry, nothing.exit, nothing.return_pct, nothing.path, nothing.notes) == ("no_data", None, None, None, [], ["no daily bar of SHEL.L is stored"])
    last = outcomes.score(plan, bars["XOM"], bars["SPY"], None, LAST, HORIZONS, symbol="XOM")
    assert (last.result, last.entry, last.return_pct, last.notes) == ("no_data", None, None, [f"no daily bar is stored after {LAST.isoformat()} yet: the future is not stored"])
    one = outcomes.score(record.plan_of(reals["be3e391cac5f"].rec), bars["MSFT"], bars["SPY"], None, date(2026, 11, 18), HORIZONS, symbol="MSFT")
    assert one.result == "open" and one.entry.date == LAST and len(one.path) == 1 and one.path[0].date == LAST


def test_a_stock_split_is_adjusted_from_the_stored_table_and_a_suspected_split_with_no_table_is_not_scored(tmp_path):
    """Netflix split 10 for 1 on 2025-11-17: the stored bars close at 1112.17 on the 14th and open at 110.75 on the 17th. NO REAL PLAN EXISTS for Netflix, so the real AMD plan's distances are put on its real bars (see the top of this file).
    With the table (read from a file in the documented shape) a decision made the day before the split and bought ON the split day reads at 1107.50, the plan's own scale, and the target of that scale ends it the next day; read raw, the same bars
    are a stop at the open and a loss of 90%; with no table the outcome is `no_data` and says why."""
    reals = {r.record_id: r for r in real_records()}
    bars = real_bars({"NFLX", "ORLY"})
    day = date(2025, 11, 14)
    net = bars["NFLX"]
    before, after = next(b for b in net if b["date"] == "2025-11-14"), next(b for b in net if b["date"] == "2025-11-17")
    assert round(before["close"] / after["open"], 1) == 10.0                                              # the table below is checked against the bars: a tenth
    (tmp_path / "NFLX.json").write_text(json.dumps({"symbol": "NFLX", "fetched_utc": "2026-11-20T18:00:00Z", "provider": "massive", "splits": [{"execution_date": "2025-11-17", "split_from": 1, "split_to": 10, "id": "x"}]}), encoding="utf-8")
    table, why = splits.read("NFLX", tmp_path)
    assert why == "" and table.symbol == "NFLX" and splits.load("NFLX", tmp_path) == table and table.splits[0].factor == 10
    assert splits.read("ORLY", tmp_path) == (None, "no split table is stored for ORLY") and splits.load("ORLY", tmp_path) is None
    (tmp_path / "ORLY.json").write_text("{not json", encoding="utf-8")
    assert splits.read("ORLY", tmp_path) == (None, "the split table of ORLY could not be read (JSONDecodeError)")
    (tmp_path / "ORLY.json").write_text(json.dumps({"symbol": "NFLX", "splits": []}), encoding="utf-8")
    assert splits.read("ORLY", tmp_path) == (None, "the split table of ORLY is for NFLX")
    plan = restated(record.plan_of(reals["83910656bece"].rec), close_on(net, "2025-11-14"), day)
    assert (plan.limit_price, plan.target, plan.stop) == (Decimal("1112.1700"), Decimal("1122.1794"), Decimal("850.1847"))
    out = outcomes.score(plan, net, bars["SPY"], None, day, HORIZONS, symbol="NFLX", splits=table)
    assert (out.result, out.entry.date.isoformat(), str(out.entry.price), out.exit.date.isoformat(), str(out.exit.price), out.exit.reason) == ("won", "2025-11-17", "1107.5000", "2025-11-18", "1122.1794", "target")
    assert out.split_adjusted and out.notes == ["adjusted for the 10-for-1 split of 2025-11-17"] and out.return_pct == 0.013255 and out.path[0].close == price_door(d(after["close"]) * 10)
    raw = simulator.follow(price_door(after["open"]), Decimal(1), "2025-11-17", net, plan.stop, plan.target, plan.time_stop_date.isoformat(), Decimal(0))
    assert (raw.exit_reason, raw.exit_date, raw.return_pct) == ("stop", "2025-11-17", 0.0) and plan.stop > price_door(after["open"])        # the plan's scale against a post-split price: stopped at the open without a loss of its own
    blind = outcomes.score(plan, net, bars["SPY"], None, day, HORIZONS, symbol="NFLX", table_missing="no split table is stored for NFLX")
    assert blind.result == "no_data" and not blind.split_adjusted and blind.entry is None and blind.notes == ["NFLX moved -90.0% from one close to the next open on 2025-11-17, which looks like a split, and no split table is stored for NFLX: it is not scored across it"]
    found = splits.suspect_move(net, day, date(2025, 12, 31), load_config().split_suspect_pct)
    assert found[0] == date(2025, 11, 17) and abs(found[1] + Decimal("0.9")) < Decimal("0.01")
    assert splits.suspect_move(net, date(2025, 11, 17), date(2025, 12, 31), load_config().split_suspect_pct) is None                    # the move is the entry's own night, not on the way
    shares, prices = table.adjust(Decimal(8), {"stop": plan.stop, "target": plan.target, "cost_basis": None}, date(2025, 11, 17))
    assert (shares, prices) == (Decimal(80), {"stop": Decimal("85.0185"), "target": Decimal("112.2179"), "cost_basis": None}) and table.adjust(Decimal(8), {}, date(2025, 11, 18)) == (Decimal(8), {})
    assert table.factor_between(date(2025, 11, 14), date(2025, 11, 20)) == 10 and table.factor_between(date(2025, 11, 17), date(2025, 11, 20)) == 1


def test_a_recommendation_is_read_as_the_record_has_it_in_schema_2_and_3_and_the_verdict_is_the_critics_or_the_advisors():
    """`plan_of` on a schema 3 record of a work order (MSFT of the quoted-theme session) and a schema 2 record of the ledger (Quick Service's), through the price door; the verdicts of three real orders: the Critic approved or denied, the Advisor's neutral
    companies the Account Manager decided on, and what nobody reviewed."""
    reals = {r.record_id: r for r in real_records()}
    msft, qsr = reals["be3e391cac5f"], reals["a497639f3127"]
    assert (msft.rec["identity"]["schema_version"], qsr.rec["identity"]["schema_version"]) == (3, 2)
    assert record.plan_of(msft.rec) == record.PlanView(entry_low=Decimal("516.2709"), entry_high=Decimal("531.4891"), entry_mid=Decimal("523.88"), limit_price=Decimal("523.88"), stop=Decimal("478.2255"), target=Decimal("528.5949"),
                                                       time_stop_date=date(2027, 1, 8), horizon_bars=40, order_style="limit")
    plan = record.plan_of(qsr.rec)
    assert (plan.stop, plan.target, plan.time_stop_date, plan.horizon_bars) == (Decimal("74.7294"), Decimal("79.5762"), date(2027, 1, 1), 40)
    sndk = record.plan_of(reals["816aea79fcda"].rec)
    assert sndk.target is None and sndk.stop == Decimal("564.6116") and record.plan_of({"symbol": "MU", "stance": "neutral"}) == record.PlanView() and record.plan_of(None) == record.PlanView()      # a capital-return plan has no target; a neutral company has no plan
    who = record.identity_of(msft.order, msft.rec, msft.day)
    assert (who["key"], who["symbol"], who["verdict"], who["critic"].approved, who["critic"].grade, who["advisor"].family) == ("2026-11-13:be3e391cac5f", "MSFT", "approved", True, who["critic"].grade, "conviction") and who["critic"].score == 81
    assert [record.verdict_of(reals[r].order, r) for r in ("937bfd681be3", "8b78a7afdbcc", "816aea79fcda", "6d8959fcf54e")] == ["denied", "approved", "denied", "denied"]     # APA (risk), XOM, SNDK (risk), PG (the market was closed)
    assert [record.verdict_of(reals["816aea79fcda"].order, d["record_id"]) for d in reals["816aea79fcda"].order.trade_decisions] == ["denied", "neutral", "neutral"]
    assert record.verdict_of(reals["816aea79fcda"].order, "not-a-record") == "unreviewed"
    assert record.caveat_sentences(["small_sample"]) == [record.CAVEATS["small_sample"]] and len(record.CAVEATS) == 11 and "fast_models" in record.CAVEATS
    with pytest.raises(ValueError):
        record.caveat_sentences(["everything_is_fine"])
    first = FIRST
    assert first == date(2025, 5, 1) and load_config().bucket_of(73) == (70, 80) and load_config().bucket_of(100) == (80, 101) and load_config().bucket_of(101) is None
