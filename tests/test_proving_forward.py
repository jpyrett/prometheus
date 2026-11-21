"""The FORWARD return of every outcome, and the family and score tables that read it (design/advisor_exit_plan_spec.md rules E5 and E6; design choices entry 139; P2 and P12) on REAL data.

A plan's own exit (`outcomes[].result`, exit by stop, target or time stop) depends on where the Advisor put the target and the stop: a target 1% over the entry makes nearly every idea a "win". The FORWARD excess (the stock's close N sessions after the decision day
against the benchmark's, from the stored bars) does not: it measures the PICK. Every outcome carries it (`forward`, 5, 10, 20 and 40 sessions) and `by_family` and `by_score` read it with their n, the standard error of the mean and the chance band.

What is real here: the stored daily bars (read only) and the recorded fast six-month oil run `tests/fixtures/proving_real/oil_weekly_6m_fast.json` with its research cache, walked again from that cache (free, no model) with the code as it is: 23 weekly decision days, 91
outcomes. Every figure is recomputed here by hand, with a loop of the test's own that calls nothing of `outcomes`, `stats` or `report_card`, in exact Decimal.
"""
import json
from datetime import date
from decimal import ROUND_HALF_UP, Context, Decimal

import pytest
from proving_helpers import real_bars, stored_bars_available
from test_proving_speed import FAST_6M, FAST_6M_INTENT, FAST_6M_REQUEST, six_month_store

from prometheus.proving import engine, outcomes, record, report_card, splits, stats
from prometheus.proving.config import CONFIG_PATH, ProvingConfig, load_config

pytestmark = pytest.mark.skipif(not stored_bars_available(), reason="the stored daily bars (data/historical/day_aggs_v1) are not on this machine")

CTX = Context(prec=60)
SESSIONS = [5, 10, 20, 40]
CONFIG = load_config()


def six(x: Decimal) -> float:
    return float(x.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))


def written(x: float) -> Decimal:
    return Decimal(str(x))


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """The recorded fast six-month oil run walked again from its cache, with no key and no model."""
    patch = pytest.MonkeyPatch()
    patch.setattr(engine, "data_stamp", lambda report: FAST_6M.coverage["stamp"])
    patch.setattr(engine, "models_used", lambda: FAST_6M.models)
    patch.setattr(engine, "versions", lambda: FAST_6M.config_versions)
    patch.setattr(engine, "spare_of", lambda config=None: 0)
    patch.setattr("prometheus.proving.bars.usable_last_day", lambda now=None: FAST_6M.range.last_bar_day)
    patch.setenv("OPENROUTER_API_KEY", "")
    try:
        yield engine.run_time_machine(FAST_6M_REQUEST, store=six_month_store(tmp_path_factory.mktemp("forward")), intent=FAST_6M_INTENT)
    finally:
        patch.undo()


def hand_forward(bars_of: dict, symbol: str, day: date, sessions: int) -> tuple[float | None, float | None, float | None]:
    """(the stock's return, SPY's, the excess) from the close of the decision day to the close `sessions` sessions later, from the stored bars: no helper of the code under test. The decision day is a trading day with a stored bar."""
    close = {s: {b["date"]: Decimal(str(b["close"])) for b in bars_of[s]} for s in (symbol, "SPY")}
    later = sorted(d for d in close[symbol] if d > day.isoformat())
    if day.isoformat() not in close[symbol] or len(later) < sessions:
        return None, None, None
    end = later[sessions - 1]
    if day.isoformat() not in close["SPY"] or end not in close["SPY"]:
        return six(close[symbol][end] / close[symbol][day.isoformat()] - 1), None, None
    stock, bench = six(close[symbol][end] / close[symbol][day.isoformat()] - 1), six(close["SPY"][end] / close["SPY"][day.isoformat()] - 1)
    return stock, bench, six(written(stock) - written(bench))


def test_the_forward_block_of_a_real_stock_is_its_stored_closes_and_spys_read_by_hand():
    """Microsoft, decision day 2026-03-25 (close 371.04): the closes 5, 10, 20 and 40 sessions later are 369.37 (2026-04-01), 373.07 (04-09), 415.75 (04-23) and 419.09 (05-21); SPY's 655.24, 679.91, 708.45 and 742.72 against 656.82."""
    bars = real_bars(["MSFT", "SPY"])
    block = outcomes.forward_of(bars["MSFT"], bars["SPY"], date(2026, 3, 25), SESSIONS, CONFIG)
    assert [f.sessions for f in block] == SESSIONS
    assert [(f.return_pct, f.benchmark_return_pct, f.excess_return_pct) for f in block] == [(-0.004501, -0.002406, -0.002095), (0.005471, 0.035154, -0.029683), (0.120499, 0.078606, 0.041893), (0.129501, 0.130782, -0.001281)]
    assert [hand_forward(bars, "MSFT", date(2026, 3, 25), n) for n in SESSIONS] == [(f.return_pct, f.benchmark_return_pct, f.excess_return_pct) for f in block]


def test_a_weekend_decision_is_measured_from_fridays_close_and_the_future_is_not_stored_yet():
    bars = real_bars(["MSFT", "SPY"])
    saturday = outcomes.forward_of(bars["MSFT"], bars["SPY"], date(2026, 3, 28), [5], CONFIG)           # a Saturday: the last close before it is Friday the 27th
    assert date(2026, 3, 28).weekday() == 5 and saturday[0].return_pct == 0.045155 and hand_forward(bars, "MSFT", date(2026, 3, 27), 5)[0] == 0.045155
    recent = outcomes.forward_of(bars["MSFT"], bars["SPY"], date(2026, 11, 10), SESSIONS, CONFIG)       # the bars end on 2026-11-19: seven sessions after the 10th
    assert recent[0].return_pct == -0.050491 and [f.return_pct for f in recent[1:]] == [None, None, None] and [f.excess_return_pct for f in recent[1:]] == [None, None, None]
    today = outcomes.forward_of(bars["MSFT"], bars["SPY"], date(2026, 11, 20), SESSIONS, CONFIG)         # a trading day whose own bar is not stored: no close to measure from
    assert [f.return_pct for f in today] == [None] * 4 and [f.sessions for f in today] == SESSIONS
    assert [f.return_pct for f in outcomes.forward_of([], None, date(2026, 3, 25), SESSIONS, CONFIG)] == [None] * 4      # no bars at all


def test_without_the_benchmark_the_stock_still_has_its_return_and_the_excess_is_left_out():
    bars = real_bars(["MSFT"])
    block = outcomes.forward_of(bars["MSFT"], None, date(2026, 3, 25), [5, 20], CONFIG)
    assert [(f.return_pct, f.benchmark_return_pct, f.excess_return_pct) for f in block] == [(-0.004501, None, None), (0.120499, None, None)]


def test_a_forward_return_across_a_real_split_is_in_the_basis_of_the_decision_day_and_with_no_table_it_is_left_out():
    """Netflix split 10 for 1 on 2025-11-17. From the close of 2025-11-13 (1,154.23) one session later is 2025-11-14 (1,112.17: -3.6%); two sessions later is the 17th, 110.29 a share, which is 1,102.90 in the basis of the 13th: -4.4%, not -90%. With NO table the
    horizon that crosses the split is None (a split cannot be told from a loss) and the one before it stands."""
    bars = real_bars(["NFLX", "SPY"])
    table = splits.SplitTable(symbol="NFLX", splits=[splits.Split(execution_date=date(2025, 11, 17), split_from=1.0, split_to=10.0)])
    with_table = outcomes.forward_of(bars["NFLX"], bars["SPY"], date(2025, 11, 13), [1, 2], CONFIG, table)
    assert [f.return_pct for f in with_table] == [-0.03644, six(Decimal("1102.9") / Decimal("1154.23") - 1)] and with_table[1].return_pct == -0.044471
    blind = outcomes.forward_of(bars["NFLX"], bars["SPY"], date(2025, 11, 13), [1, 2], CONFIG, None)
    assert [f.return_pct for f in blind] == [-0.03644, None] and [f.excess_return_pct for f in blind][1] is None


def test_every_outcome_of_a_real_run_has_the_four_horizons_and_each_is_what_the_stored_bars_say_by_hand(run):
    """91 outcomes, traded or not, of every verdict and result: each one has a forward block of 5, 10, 20 and 40 sessions, whatever its plan did, and every figure of it is recomputed here from the stored bars."""
    symbols = {o.symbol for o in run.outcomes}
    bars = real_bars(symbols)
    assert len(run.outcomes) == 91 and all([f.sessions for f in o.forward] == SESSIONS for o in run.outcomes)
    checked = 0
    for o in run.outcomes:
        table = splits.load(o.symbol)
        assert table is None or not any(s.execution_date >= date(2025, 5, 1) for s in table.splits), o.symbol      # no stock of this run split while the bars are stored: the by-hand reading needs no basis
        for f in o.forward:
            assert (f.return_pct, f.benchmark_return_pct, f.excess_return_pct) == hand_forward(bars, o.symbol, o.decision_day, f.sessions), (o.key, f.sessions)
            checked += f.return_pct is not None
    assert checked > 250 and any(f.return_pct is None for o in run.outcomes for f in o.forward)             # most horizons are read, and the 40-session one of the last weeks is not stored yet
    not_filled = [o for o in run.outcomes if o.result in ("not_filled", "no_data")]
    assert all(any(f.return_pct is not None for f in o.forward) for o in not_filled if o.result == "not_filled")      # the pick of an order that was not placed is judged too


def test_the_forward_return_does_not_depend_on_the_plan_the_entry_or_the_fill(run):
    """The same recommendation scored with a plan that has nothing in it (no stop, no target, no time stop: it can never close) has the same forward block as with its own plan."""
    bars = real_bars({o.symbol for o in run.outcomes})
    for o in [x for x in run.outcomes if x.result in ("won", "lost")][:6]:
        bare = outcomes.score(record.PlanView(), bars[o.symbol], bars["SPY"], None, o.decision_day, [5], symbol=o.symbol)
        assert bare.result == "open" and bare.forward == o.forward and o.exit is not None


def test_the_family_table_reads_the_forward_excess_with_n_se_and_the_chance_band_and_has_a_neutral_row(run):
    """Every row, by hand: n (the outcomes that have the figure, of every verdict), the mean, the standard error of the mean (the sample variance over n), `inside_chance` (the mean is within 2 standard errors of zero; None under two) and the small-sample mark
    (under 20). A company the Advisor scored neutral has no family and is the `neutral` row, last."""
    by_name = {}
    for o in run.outcomes:
        by_name.setdefault((o.advisor.family if o.advisor else None) or "neutral", []).append(o)
    assert [r.family for r in run.stats.by_family] == [*sorted(n for n in by_name if n != "neutral"), "neutral"] and set(by_name) == {"conviction", "trend", "neutral"}
    for row in run.stats.by_family:
        assert [f.sessions for f in row.forward] == SESSIONS
        for f in row.forward:
            values = [written(x.excess_return_pct) for o in by_name[row.family] for x in o.forward if x.sessions == f.sessions and x.excess_return_pct is not None]
            assert f.n == len(values) and f.small_sample == (len(values) < 20)
            if not values:
                assert (f.mean_excess_pct, f.se_pct, f.inside_chance) == (None, None, None)
                continue
            mean = CTX.divide(sum(values, Decimal(0)), Decimal(len(values)))
            variance = CTX.divide(CTX.divide(sum(((v - mean) ** 2 for v in values), Decimal(0)), Decimal(len(values) - 1)), Decimal(len(values))) if len(values) > 1 else Decimal(0)
            error = variance.sqrt(CTX)
            assert (f.mean_excess_pct, f.se_pct) == (six(mean), six(error)) and f.inside_chance == (None if len(values) < 2 else abs(mean) <= 2 * error)
    trend = next(r for r in run.stats.by_family if r.family == "trend")
    twenty = next(f for f in trend.forward if f.sessions == 20)
    assert (twenty.n, twenty.mean_excess_pct, twenty.inside_chance, twenty.small_sample) == (29, 0.031894, False, False)      # the recorded run's trend picks: +3.2 points after 20 sessions, 2.4 standard errors up


def test_the_score_table_reads_the_forward_excess_of_the_strategy_family_picks_in_the_configs_buckets(run):
    """The buckets are <60, 60-65, 65-70, 70-75 and 75+ (the config); a neutral company is not in them (it has no family: the family table's `neutral` row). Each bucket is recomputed by hand like a family row."""
    assert [(b.low, b.high) for b in run.stats.by_score] == [(0, 60), (60, 65), (65, 70), (70, 75), (75, 101)]
    for bucket in run.stats.by_score:
        mine = [o for o in run.outcomes if o.advisor and o.advisor.family and o.advisor.score is not None and bucket.low <= o.advisor.score < bucket.high]
        for f in bucket.forward:
            values = [written(x.excess_return_pct) for o in mine for x in o.forward if x.sessions == f.sessions and x.excess_return_pct is not None]
            assert f.n == len(values) and (f.mean_excess_pct == (six(CTX.divide(sum(values, Decimal(0)), Decimal(len(values)))) if values else None))
    assert sum(next(f for f in b.forward if f.sessions == 20).n for b in run.stats.by_score) == next(f for r in run.stats.by_family if r.family == "conviction" for f in r.forward if f.sessions == 20).n + next(f for r in run.stats.by_family if r.family == "trend" for f in r.forward if f.sessions == 20).n


def test_the_win_rate_stays_the_plans_beside_the_forward_excess(run):
    """`win_rate` is what it was: won over closed BY THE PLAN. The family's trend picks closed 28 times and won 19 of them (68%) while the same picks' forward excess after 20 sessions is +3.2 points: two different measures, both on the row."""
    trend = next(r for r in run.stats.by_family if r.family == "trend")
    closed = [o for o in run.outcomes if o.advisor and o.advisor.family == "trend" and o.result in ("won", "lost") and o.excess_return_pct is not None]
    assert (trend.closed, trend.win_rate) == (len(closed), six(Decimal(sum(o.result == "won" for o in closed)) / Decimal(len(closed))))
    assert trend.mean_excess_pct == six(sum((written(o.excess_return_pct) for o in run.outcomes if o.advisor and o.advisor.family == "trend" and o.excess_return_pct is not None), Decimal(0)) / Decimal(trend.n))


def test_a_trailing_stop_exit_is_counted_by_the_exit_mix_and_a_won_one_is_a_win():
    """`Stats.outcomes.exits.trailing_stop` counts the closes by a trailing stop: the real Microsoft trailing plan (see test_trailing_stop.py) is one, and it won."""
    from test_trailing_stop import real_plan
    bars = real_bars(["MSFT", "SPY"])
    out = outcomes.score(real_plan(), bars["MSFT"], bars["SPY"], None, date(2026, 3, 25), [5, 10, 20], symbol="MSFT")
    result = stats.compute([out], [], CONFIG)
    assert (result.outcomes.exits.trailing_stop, result.outcomes.exits.stop, result.outcomes.exits.target, result.outcomes.won, result.outcomes.win_rate) == (1, 0, 0, 1, 1.0)
    assert [f.family for f in result.by_family] == [] and [(f.sessions, f.n) for f in stats.forward_stats([out], CONFIG)] == [(5, 1), (10, 1), (20, 1), (40, 1)]


def test_the_critics_report_card_carries_the_forward_excess_of_each_family_of_the_denied_and_the_approved(run):
    card = report_card.build([run], folders=[], include_track=False)
    version = card["by_config_version"][card["config_versions"][-1]]
    assert version["by_family"], "the run has reviewed recommendations of strategy families"
    entries, _ = report_card.entries_of([run], report_card.ReviewIndex([run], []))
    for row in version["by_family"]:
        forward = row["forward"]
        assert set(forward) == {"all", "denied", "approved"}
        for who, kind in (("all", None), ("denied", "denied"), ("approved", "approved")):
            mine = [e for e in entries if e.family == row["family"] and (kind is None or e.verdict == kind) and not e.suitability]
            for cell in forward[who]:
                values = [written(e.forward[cell["sessions"]]) for e in mine if e.forward.get(cell["sessions"]) is not None]
                assert cell["n"] == len(values) and cell["mean_excess_pct"] == (six(CTX.divide(sum(values, Decimal(0)), Decimal(len(values)))) if values else None)
    assert "forward excess, all" in report_card.report(card) and any(cell["n"] for row in version["by_family"] for cell in row["forward"]["all"])


def test_the_forward_sessions_and_the_buckets_are_checked_when_the_config_is_read():
    raw = {k: v for k, v in json.loads(CONFIG_PATH.read_text(encoding="utf-8")).items() if not k.startswith("_")}
    assert ProvingConfig.model_validate(raw).forward_sessions == SESSIONS and ProvingConfig.model_validate(raw).score_buckets == [0, 60, 65, 70, 75, 101]
    for bad in ([], [5, 5], [10, 5], [0, 5]):
        with pytest.raises(ValueError, match="whole numbers of sessions"):
            ProvingConfig.model_validate({**raw, "forward_sessions": bad})
