"""Critic v3, section 4 (design/critic_v3_two_reasons_spec.md): eight more chart patterns, calculated in code from the STORED day bars, and the net points as a bonus on the score.

Every pattern is tested on REAL stored bars (`data/historical/day_aggs_v1`, read only, through the Data Clerk's own metric `close_history`): a real (symbol, day) where it fires and one where it does not. HOW THEY WERE FOUND: a
throwaway script (kept in the scratchpad, not in the project) walked every symbol whose news is stored (`news/<SYMBOL>.txt`) over the stored bars, built `close_history` for each day exactly as the Data Clerk does and listed where each
pattern fires. THE RECENCY RULE (`chart.pattern_recent_sessions`, 10: a neckline pattern counts only while its neckline was broken within the last 10 sessions, a triangle only while its newest swing point is within the last 20) cut the
firing rate on the last 45 sessions of 50 symbols (2,250 stock-days) from 931 to 228 for `double_bottom` (41.4% to 10.1%), 643 to 189 for `double_top` (28.6% to 8.4%), 8 to 0 for `inverse_head_and_shoulders`, 75 to 8 for
`head_and_shoulders`, 123 to 90 for `ascending_triangle` and 25 to 23 for `descending_triangle`; the two 52-week patterns are the last close itself and are as they were (141 and 63). Over every day that has the 120 sessions behind it
(13,700 stock-days, 7,050 for the 52-week ones): `double_bottom` 38.7% to 8.3%, `double_top` 24.3% to 6.0%, `inverse_head_and_shoulders` 1.1% to 0.2%, `head_and_shoulders` 2.2% to 0.4%, `ascending_triangle` 3.3% to 3.0%,
`descending_triangle` 2.6% to 2.4%, `breakout_52w` 5.6%, `breakdown_52w` 2.3%. The pairs below are, for each pattern, a day it fires on, a day of the same stock it does not fire on, and a STALE day: one on which the pattern was complete
before but too long ago (it fired without the recency rule and does not with it). No bar is invented here. The swing points a pattern names, and the session the neckline broke, are checked against the stored closes by plain Python in this
file, not by the code under test.
"""
from datetime import date

import pytest
from critic_helpers import CONFIG, captured, rec_of, review

from prometheus.agents.critic import chart as ch
from prometheus.agents.critic import patterns as pt
from prometheus.agents.data_clerk import metrics_catalog as catalog
from prometheus.agents.data_clerk.bars import BarStore
from prometheus.common import trading_clock as tc
from prometheus.rag.config import REPO_ROOT

BARS = REPO_ROOT / "data" / "historical" / "day_aggs_v1"
C = CONFIG.chart
pytestmark = pytest.mark.skipif(not BARS.is_dir(), reason="the stored day bars are not on this machine")

# pattern: (symbol, a day it fires on, a day of the same symbol it does not fire on, a STALE day: (symbol, day) where it fired before the recency rule and does not now, or None)
PAIRS = {
    "double_bottom": ("AAPL", "2026-09-25", "2026-10-20", ("AAPL", "2026-10-15")),
    "double_top": ("AMD", "2026-07-17", "2026-10-29", ("AMD", "2026-08-24")),
    "inverse_head_and_shoulders": ("DPZ", "2026-07-20", "2026-10-30", ("DPZ", "2026-08-14")),
    "head_and_shoulders": ("YUM", "2026-08-10", "2026-10-30", ("YUM", "2026-09-22")),
    "ascending_triangle": ("AMZN", "2026-10-16", "2026-10-15", ("KMB", "2026-09-22")),
    "descending_triangle": ("MCD", "2026-09-21", "2026-10-30", ("MCD", "2026-10-12")),
    "breakout_52w": ("AAPL", "2026-09-25", "2026-10-30", None),
    "breakdown_52w": ("GIS", "2026-09-30", "2026-10-30", None),
}
NO_RECENCY = C.model_copy(update={"pattern_recent_sessions": 10**6})      # the same thresholds with the recency rule made so wide that it never binds: what the patterns were before it


def closes(symbol: str, day: str) -> list[list]:
    """The Data Clerk's `close_history` of a symbol as of a day, from the stored bars and the stored split tables (`DataClerkTools.metrics_for`, the code path of `request_metrics`): [[date, close], ...] oldest first, in the basis of the day."""
    from prometheus.tools.data_clerk import DataClerkTools
    return DataClerkTools().metrics_for([symbol], [{"name": "close_history"}], date.fromisoformat(day))["results"][symbol]["close_history"]["value"]


def stored(symbol: str, day: str) -> list[list]:
    """The same closes as the day files and the REST bars hold them: UNADJUSTED, [[date, close], ...] oldest first (the 300 newest on or before the day)."""
    return [[b.date.isoformat(), b.close] for b in BarStore(BARS).load([symbol], end=date.fromisoformat(day)).bars(symbol)]


def judge(name: str, symbol: str, day: str):
    return pt.JUDGES[name]({"close_history": closes(symbol, day)}, C)


def test_the_eight_patterns_are_the_specs_and_chart_judges_them_with_the_old_fifteen():
    assert set(pt.JUDGES) == {"double_bottom", "double_top", "inverse_head_and_shoulders", "head_and_shoulders", "ascending_triangle", "descending_triangle", "breakout_52w", "breakdown_52w"} == set(PAIRS)
    assert set(pt.JUDGES) <= set(ch.JUDGES) and len(ch.JUDGES) == len(C.patterns) == 23
    assert (C.pivot_window, C.pattern_lookback, C.pattern_recent_sessions, C.level_tolerance, C.min_gap, C.min_depth, C.year_sessions) == (5, 120, 10, 0.03, 10, 0.05, 252)


@pytest.mark.parametrize("name", sorted(PAIRS))
def test_each_pattern_fires_on_a_real_stock_day_and_not_on_another_day_of_the_same_stock(name):
    symbol, fires, quiet, _ = PAIRS[name]
    held, text = judge(name, symbol, fires)
    assert held is True and text, (name, text)
    held, text = judge(name, symbol, quiet)
    assert held is False and text, (name, text)


@pytest.mark.parametrize("name", [n for n, p in sorted(PAIRS.items()) if p[3] is not None])
def test_a_pattern_that_was_complete_too_long_ago_does_not_count_though_it_did_before_the_recency_rule(name):
    """Critic v3 recency: on the stale day the same real stock showed the pattern (it fires with the recency rule made so wide that it never binds) but its neckline was broken more than 10 sessions ago (a triangle: its newest swing point is
    more than 20 sessions old), so it does not count. The same stock on its firing day fires under both."""
    symbol, fires, _, (stale_symbol, stale) = PAIRS[name]
    assert pt.JUDGES[name]({"close_history": closes(symbol, fires)}, NO_RECENCY)[0] is True
    assert pt.JUDGES[name]({"close_history": closes(stale_symbol, stale)}, NO_RECENCY)[0] is True
    held, text = judge(name, stale_symbol, stale)
    assert held is False and ("last 10 sessions" in text or "older than 20" in text), (name, text)


@pytest.mark.parametrize("name, above", [("double_bottom", True), ("double_top", False), ("inverse_head_and_shoulders", True), ("head_and_shoulders", False)])
def test_the_neckline_break_a_pattern_names_is_the_first_close_beyond_it_after_the_last_swing_point_and_within_ten_sessions(name, above):
    """Checked by plain Python on the stored closes: the evidence names the neckline and the session of its break; that session is the FIRST close beyond the neckline after the pattern's last swing point, it is at most 10 sessions ago,
    and the last close is still beyond the neckline."""
    import re
    symbol, day, _, _ = PAIRS[name]
    rows = closes(symbol, day)
    held, text = judge(name, symbol, day)
    assert held
    neck = float(re.search(r"(?:a high of|a low of|the neckline of) ([0-9]+\.[0-9]{2})", text).group(1))
    broke_day = re.search(r"broken (?:today|\d+ sessions ago) \((\d{4}-\d{2}-\d{2})\)", text).group(1)
    dates = [d for d, _ in rows]
    values = [c for _, c in rows]
    beyond = (lambda x: x > neck) if above else (lambda x: x < neck)
    last_pivot = dates.index(re.findall(r"([0-9]+\.[0-9]{2}) \((\d{4}-\d{2}-\d{2})", text)[-1][1])
    broke = dates.index(broke_day)
    assert beyond(values[broke]) and not any(beyond(v) for v in values[last_pivot + 1:broke]) and beyond(values[-1])
    assert len(rows) - broke <= C.pattern_recent_sessions and broke > last_pivot


def test_the_close_history_is_the_data_clerks_real_closes_oldest_first_and_the_last_is_the_days_close():
    rows = closes("AAPL", "2026-09-25")
    assert len(rows) == 300 and rows[-1][0] == "2026-09-25" and all(rows[i][0] < rows[i + 1][0] for i in range(len(rows) - 1)) and all(isinstance(c, float) and c > 0 for _, c in rows)
    store = BarStore(BARS).load(["AAPL"], end=date(2026, 9, 25))
    assert [c for _, c in rows] == [b.close for b in store.bars("AAPL")][-300:] and rows[-1][1] == store.bars("AAPL")[-1].close == 341.07


def test_while_a_day_is_bound_the_close_history_stops_at_that_day_whatever_date_is_asked_for():
    """Nothing from the future (P2): the metric is made by `metrics_for`, which cuts the bars at the bound day, so the patterns of a Time Machine day never see a later close."""
    from prometheus.tools.data_clerk import DataClerkTools
    tools = DataClerkTools()
    unbound = tools.metrics_for(["AAPL"], [{"name": "close_history"}], date(2026, 11, 19))["results"]["AAPL"]["close_history"]["value"]
    with tc.as_of(date(2026, 9, 25)):
        bound = tools.metrics_for(["AAPL"], [{"name": "close_history"}], date(2026, 11, 19))["results"]["AAPL"]["close_history"]["value"]
    assert unbound[-1][0] > "2026-11-01" and bound[-1] == ["2026-09-25", 341.07] and bound == closes("AAPL", "2026-09-25") and max(d for d, _ in bound) == "2026-09-25"


# ---------------------------------------------------------------- splits: the closes are in the basis of the last day

def test_a_real_split_inside_the_window_is_put_into_the_closes_in_the_basis_of_the_last_day():
    """SanDisk's REAL 8-for-1 split, executed 2026-11-20 (`data/historical/splits_v1/SNDK.json`, written from Massive's table; the stored bars are UNADJUSTED): the stored closes fall from 755.16 to 79.24 in a night, a crash of
    89.5% that never was. `close_history` divides every close before the split by its factor (the Proving Ground's `SplitTable.factor_between`, through the price door), so the day before is 94.395, the last close is as stored and the
    series is continuous. The metric says which split it applied. A pattern read off the raw series is a lie (a 52-week breakdown); read off the adjusted one it is not there."""
    from prometheus.tools.data_clerk import DataClerkTools
    raw, done = stored("SNDK", "2026-11-20"), closes("SNDK", "2026-11-20")
    assert raw[-2:] == [["2026-11-19", 755.16], ["2026-11-20", 79.24]] and abs(raw[-1][1] / raw[-2][1] - 1) > 0.89            # as stored: -89.5% in one night
    assert [d for d, _ in done] == [d for d, _ in raw] and done[-1] == raw[-1] and done[-2] == ["2026-11-19", 94.395]
    assert all(abs(c - r / 8) <= 0.00005 + 1e-9 for (_, c), (_, r) in zip(done[:-1], raw[:-1]))                                              # every earlier close is the stored one over 8, to the price door's four places
    assert abs(done[-1][1] / done[-2][1] - 1) < 0.17 and max(abs(done[i + 1][1] / done[i][1] - 1) for i in range(len(done) - 1)) < 0.5      # no night of the whole window moves like a split any more
    summary = DataClerkTools().metrics_for(["SNDK"], [{"name": "close_history"}], date(2026, 11, 20))["results"]["SNDK"]["close_history"]["inputs_summary"]
    assert summary["splits"] == [{"date": "2026-11-20", "factor": 8.0}] and summary["last_close"] == 79.24
    broke_raw, text = pt.breakdown_52w({"close_history": raw}, C)
    assert broke_raw is True and "79.24" in text and "194.38" in text                                                                 # unadjusted: under the lowest close of the year before it (194.38)
    held, text = pt.breakdown_52w({"close_history": done}, C)
    assert held is False and "79.24" in text and "24.30" in text                                                                      # adjusted: the year's lowest close was 24.30 in today's shares


def test_a_split_after_the_bound_day_is_never_applied_and_a_stock_with_no_split_in_the_window_is_as_stored():
    """Nothing from the future (P2): SNDK's table holds the split of 2026-11-20, but the desk working as of 2026-11-19 (or asked for that day) has not seen it, so its closes are the stored ones and the metric names no split. AAPL's table
    holds five splits, the newest in 2020: none is inside the window, so its closes are exactly the stored ones. A symbol with no table is not adjusted."""
    from prometheus.tools.data_clerk import DataClerkTools
    assert closes("SNDK", "2026-11-19") == stored("SNDK", "2026-11-19") and closes("SNDK", "2026-11-19")[-1] == ["2026-11-19", 755.16]
    with tc.as_of(date(2026, 11, 19)):
        bound = DataClerkTools().metrics_for(["SNDK"], [{"name": "close_history"}], date(2026, 11, 20))["results"]["SNDK"]["close_history"]
    assert bound["value"] == stored("SNDK", "2026-11-19") and bound["inputs_summary"]["splits"] == [] and bound["value"][-1][0] == "2026-11-19"
    assert closes("AAPL", "2026-10-30") == stored("AAPL", "2026-10-30")
    day = date(2026, 11, 20)
    no_table = catalog.compute(["SNDK"], [{"name": "close_history"}], BarStore(BARS).load(["SNDK"], end=day), {}, {}, day, None, {})["SNDK"]["close_history"]
    assert no_table["value"] == stored("SNDK", "2026-11-20") and no_table["inputs_summary"]["splits"] == []


def test_the_two_52_week_patterns_are_the_last_close_against_the_year_before_it_worked_by_hand():
    for name, symbol, day, high in (("breakout_52w", "AAPL", "2026-09-25", True), ("breakdown_52w", "GIS", "2026-09-30", False)):
        values = [c for _, c in closes(symbol, day)]
        last, before = values[-1], values[-253:-1]
        assert len(before) == 252 and (last > max(before) if high else last < min(before))
        held, text = judge(name, symbol, day)
        assert held and f"{last:.2f}" in text and f"{(max(before) if high else min(before)):.2f}" in text
    values = [c for _, c in closes("AAPL", "2026-10-30")]
    assert not values[-1] > max(values[-253:-1])                                                      # the quiet day: under the year's high


@pytest.mark.parametrize("name, kind", [("double_bottom", "low"), ("double_top", "high"), ("inverse_head_and_shoulders", "low"), ("head_and_shoulders", "high")])
def test_the_swing_points_a_pattern_names_are_real_swing_points_of_the_stored_closes(name, kind):
    """Each date the evidence names is the lowest (highest) close of the 11 closes around it (5 each side): checked by plain Python on the stored closes, not by `swing_points`."""
    import re
    symbol, day, _, _ = PAIRS[name]
    rows = closes(symbol, day)
    held, text = judge(name, symbol, day)
    assert held
    named = re.findall(r"([0-9]+\.[0-9]{2}) \((\d{4}-\d{2}-\d{2})", text)
    assert len(named) in (2, 3)
    dates = [d for d, _ in rows]
    for level, when in named:
        i = dates.index(when)
        around = [c for _, c in rows[i - 5:i + 6]]
        assert f"{rows[i][1]:.2f}" == level and (rows[i][1] == min(around) if kind == "low" else rows[i][1] == max(around)), (name, when)
    assert rows[-1][1] == [c for d, c in rows if d == day][0]
    first, last = dates.index(named[0][1]), dates.index(named[-1][1])
    assert last - first >= C.min_gap and last <= len(rows) - 1 - C.pivot_window                      # a swing point is only confirmed after five more sessions


def test_the_swing_points_of_a_small_series_and_the_level_rule():
    series = [5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 3, 4, 5, 6, 7, 8, 9]
    lows, highs = pt.swing_points([float(x) for x in series], 3)
    assert lows == [(4, 1.0), (15, 2.0)] and highs == [(10, 7.0)]                                     # the first and the last three sessions can be neither: nothing confirms them yet
    assert pt.swing_points([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 2) == ([], [])                      # a flat run has no swing
    assert pt.swing_points([3.0, 2.0, 1.0, 1.0, 2.0, 3.0], 2) == ([(2, 1.0)], [])                     # of two equal lows the first counts
    assert pt.same_level(100.0, 103.0, 0.03) and not pt.same_level(100.0, 103.1, 0.03) and pt.same_level(103.0, 100.0, 0.03)


def test_within_the_last_ten_sessions_counts_the_last_close_as_the_first_and_the_break_is_the_first_close_beyond_the_neckline():
    """The bound of the recency rule on INPUTS for the rule (not market data): of 20 closes the last 10 are indexes 10 to 19. A neckline of 100 first broken at index 10 is recent, at index 9 is not; one broken and lost again is not (the
    last close must still be beyond it); one broken before the pattern's last swing point is not a break of that pattern."""
    def series(first_beyond, last=105.0):
        values = [90.0] * 20
        for k in range(first_beyond, 20):
            values[k] = 105.0
        values[19] = last
        return values
    assert pt._within(10, 20, 10) and not pt._within(9, 20, 10) and pt._within(19, 20, 1) and not pt._within(18, 20, 1)
    assert pt._broke_recently(series(10), 100.0, True, 3, C) == 10 and pt._broke_recently(series(9), 100.0, True, 3, C) is None
    assert pt._broke_recently(series(19), 100.0, True, 3, C) == 19 and pt._broke_recently(series(10, last=95.0), 100.0, True, 3, C) is None
    assert pt._broke_recently([110.0 - x for x in series(10)], 100.0, False, 3, C) is None                         # the mirror: a close above the neckline is no downward break
    assert pt._broke_recently([200.0 - x for x in series(10)], 100.0, False, 3, C) == 10                          # 95s after the break: under the neckline from index 10
    assert pt._broke_recently(series(5), 100.0, True, 6, C) is None                                               # the first close beyond the neckline AFTER the last swing point (index 6) is index 7: 13 sessions ago


def test_a_pattern_is_not_judged_without_its_closes_never_guessed():
    short = [["2026-01-01", 10.0 + i % 3] for i in range(10)]
    for name, judge_ in pt.JUDGES.items():
        assert judge_({"close_history": short}, C) == (None, ""), name
        assert judge_({"close_history": None}, C) == (None, "") and judge_({}, C) == (None, ""), name
    year_short = closes("AAPL", "2026-09-25")[-250:]
    assert pt.breakout_52w({"close_history": year_short}, C) == (None, "") and pt.breakdown_52w({"close_history": year_short}, C) == (None, "")      # 253 closes are needed for the year before the last
    assert pt.double_bottom({"close_history": year_short}, C)[0] is not None                            # the swing patterns need only the 120 sessions


# ---------------------------------------------------------------- inside the chart and the score

def test_the_chart_counts_the_new_patterns_in_its_points_its_score_and_its_findings_and_keeps_the_series_out_of_the_review():
    rec = rec_of(captured("PG_open"))
    base = {name: {"value": value} for name, value in dict(last_close=100.0, sma_50=100.0, sma_200=100.0, pct_of_52w_high=0.5, sma_50_vs_200=0.0, cross_50_200_20d=0, structure_63d="mixed", breakout_20d=False,
                                                           volume_ratio_1d=1.0, macd_hist=0.0, rsi_14=40.0, obv_20d_change=0.0, decline_from_peak_63d=0.0, rebound_from_trough_63d=0.0, sessions_since_trough_63d=0).items()}
    up = ch.assess(rec, {**base, "close_history": {"value": closes("AAPL", "2026-09-25")}}, C)
    assert {h.name for h in up.detail.patterns} >= {"double_bottom", "breakout_52w"} and up.detail.points == sum(h.points for h in up.detail.patterns) and up.score == 50 + up.detail.points
    assert "close_history" not in up.detail.facts and all(h.direction == "bullish" for h in up.detail.patterns if h.name in ("double_bottom", "breakout_52w"))
    assert next(h for h in up.detail.patterns if h.name == "breakout_52w").points == 10 and next(h for h in up.detail.patterns if h.name == "double_bottom").evidence.startswith("two lows of 321.66")
    down = ch.assess(rec, {**base, "close_history": {"value": closes("YUM", "2026-08-10")}}, C)
    assert "head_and_shoulders" in [h.name for h in down.detail.patterns] and next(h for h in down.detail.patterns if h.name == "head_and_shoulders").points == -10
    unknown = ch.assess(rec, base, C)                                                                  # no series: the eight are not judged, and the old fifteen are as before
    assert set(pt.JUDGES) <= set(unknown.detail.unknown) and unknown.detail.points == 0


def test_chart_bearish_fires_at_the_net_of_minus_twenty_and_not_above():
    rec = rec_of(captured("PG_open"))
    base = {name: {"value": value} for name, value in dict(last_close=80.0, sma_50=90.0, sma_200=100.0, pct_of_52w_high=0.5, sma_50_vs_200=0.0, cross_50_200_20d=0, structure_63d="down", breakout_20d=False,
                                                           volume_ratio_1d=1.0, macd_hist=0.0, rsi_14=40.0, obv_20d_change=0.0, decline_from_peak_63d=0.0, rebound_from_trough_63d=0.0, sessions_since_trough_63d=0).items()}
    at = ch.assess(rec, base, C)                                                                       # bear alignment -10 and downtrend structure -10: exactly -20
    assert at.detail.points == -20 and "chart_bearish" in [f.code for f in at.findings] and next(f for f in at.findings if f.code == "chart_bearish").severity == "warning"
    above = ch.assess(rec, {**base, "structure_63d": {"value": "mixed"}}, C)
    assert above.detail.points == -10 and "chart_bearish" not in [f.code for f in above.findings]


def test_the_pattern_bonus_is_a_quarter_point_per_chart_point_capped_at_eight_either_way():
    b = C.bonus
    assert (b.per_point, b.max_points) == (0.25, 8)
    assert ch.pattern_bonus(25, b) == 6.25 and ch.pattern_bonus(-20, b) == -5.0 and ch.pattern_bonus(10, b) == 2.5 and ch.pattern_bonus(0, b) == 0.0 and ch.pattern_bonus(None, b) == 0.0
    assert ch.pattern_bonus(32, b) == 8.0 and ch.pattern_bonus(200, b) == 8.0 and ch.pattern_bonus(-32, b) == -8.0 and ch.pattern_bonus(-300, b) == -8.0
    assert ch.pattern_bonus(31, b) == 7.75 and ch.pattern_bonus(-31, b) == -7.75                       # the cap is at 32 points


def test_a_bullish_chart_raises_the_reviews_score_and_a_bearish_one_lowers_it_by_the_bonus():
    """Two REAL reviews. PG_open's own chart (near its high, MACD positive, a healthy RSI, rising volume: net +25) earns +6.25; the real SNDK review's chart (MACD negative, falling volume: net -10) earns -2.5. In both the score
    is `round(sum of the points over the weights + the bonuses)` and nothing else moved. (SNDK is denied for its risk, a fault: the score before the cap is what shows the bonus.)"""
    def base_of(r):
        applicable = [d for d in r["dimensions"].values() if d["applicable"]]
        return sum(d["points"] for d in applicable) / sum(d["weight"] for d in applicable) * 100
    plain = review(captured("PG_open"), own={})["PG"]
    assert plain["pattern_bonus"] == 6.25 and plain["chart"]["points"] == 25 and plain["score"] == round(base_of(plain) + plain["bonus"] + 6.25) > round(base_of(plain))
    bear = review(captured("SNDK"), own={})["SNDK"]
    assert bear["chart"]["points"] == -10 and bear["pattern_bonus"] == -2.5 and bear["bonus"] == 0.0 and [p["name"] for p in bear["chart"]["patterns"]] == ["macd_negative", "obv_falling"]
    assert bear["score_before_blockers"] == round(base_of(bear) - 2.5) < round(base_of(bear)) + 1 and "risk_above_profile" in bear["denial_reasons"]


def test_a_review_with_real_closes_scores_the_swing_patterns_and_the_bonus_follows_the_net():
    """The real PG_open review given the real closes of PG as of its day: whatever patterns hold, the review's bonus is the net times 0.25 capped, and the series is not in the review."""
    rows = closes("PG", "2026-11-05")
    r = review(captured("PG_open"), own=lambda symbol, last: {"close_history": rows})["PG"]
    net = r["chart"]["points"]
    assert r["pattern_bonus"] == max(-8.0, min(8.0, round(net * 0.25, 2))) and "close_history" not in r["chart"]["facts"] and len(str(r)) < 200_000
    held = {p["name"] for p in r["chart"]["patterns"]}
    for name in pt.JUDGES:
        assert (name in held) == (pt.JUDGES[name]({"close_history": rows}, C)[0] is True), name
    assert [m for m in r["dimensions"]["chart"]["measures"] if m["name"] in pt.JUDGES and m["available"]]            # the eight are measures of the chart dimension, judged from the real series


def test_the_critic_asks_for_the_close_history_and_the_63_session_return_with_its_other_numbers_within_its_cap():
    from prometheus.agents.critic import recheck
    for name in ("PG_open", "SNDK"):
        rec = rec_of(captured(name))
        names, _ = recheck.key_metrics(rec, CONFIG.limits.max_metrics)
        full, _ = recheck.key_metrics(rec, 10_000)
        assert "close_history" in names and "ret_63d" in names and "ret_5d" in names and names == full and len(names) < CONFIG.limits.max_metrics
    assert "close_history" in recheck.OWN and "ret_63d" in recheck.OWN and "ret_63d" not in recheck.ASKED_ONLY      # a name the record holds too is compared as before; `ret_5d` alone is never compared


def test_the_critic_reads_no_bar_store_and_never_asks_for_a_range_of_bars():
    import re
    for path in (REPO_ROOT / "prometheus" / "agents" / "critic").glob("*.py"):
        assert not re.search(r"request_bars|daily_bars|BarStore", path.read_text(encoding="utf-8")), path
