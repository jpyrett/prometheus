"""Tests for prometheus.agents.advisor.ranking (Advisor v1 spec sections 9.5 to 9.8 and 12, phase 5): one vote per family, the min rule, the cost, the stance, the regime, the holdings
pass, the cooldown, the sleeves, diversity and a deterministic order. Numbers are literals that put a rule at its threshold; they are not market data."""
import pytest
from advisor_helpers import CONFIG, TODAY, fr, healthy, metric, metrics_of, scored

from prometheus.agents.advisor import ranking as rk
from prometheus.agents.advisor.record import Haircut, Quality

SPY_TREND = metrics_of(spy_above_sma_200=True, spy_vol_21d_ann=0.15, spy_drawdown_63d=-0.03, spy_ret_21d=0.02, breadth_200=None)


# ---------------------------------------------------------------- the regime

def test_the_regime_is_unknown_when_spy_has_too_few_bars_and_says_why():
    state, note = rk.regime_of(metrics_of(spy_above_sma_200=None), CONFIG)
    assert state == "unknown" and note == "needs bars for spy_above_sma_200; regime not computable"
    assert rk.regime_of({}, CONFIG)[0] == "unknown" and rk.regime_of(None, CONFIG)[1] == "SPY metrics are missing; regime not computable"


@pytest.mark.parametrize("override, state", [
    ({}, "trend"), (dict(breadth_200=0.55), "trend"), (dict(breadth_200=0.45), "trend"), (dict(breadth_200=0.39), "narrow"),
    (dict(spy_vol_21d_ann=0.26), "stress"), (dict(spy_above_sma_200=False), "stress"),
    (dict(spy_above_sma_200=False, spy_drawdown_63d=-0.11, spy_ret_21d=0.06), "recovering"),         # recovering is tested before stress: it is the more specific
    (dict(spy_drawdown_63d=-0.11, spy_ret_21d=0.05), "trend"),                                         # exactly +5% is not "over 5%"
])
def test_the_regime_states(override, state):
    values = dict(spy_above_sma_200=True, spy_vol_21d_ann=0.15, spy_drawdown_63d=-0.03, spy_ret_21d=0.02, breadth_200=None)
    assert rk.regime_of(metrics_of(**{**values, **override}), CONFIG)[0] == state


# ---------------------------------------------------------------- one symbol


def test_one_vote_per_family_so_two_families_do_not_add_and_the_best_bullish_edge_leads():
    assert rk.combine([fr("trend", edge=0.004), fr("reversion", edge=0.003)]) == (0.004, False)
    assert rk.combine([fr("trend", edge=0.004), fr("event", "bearish", edge=-0.0015)]) == (pytest.approx(0.0025), True)    # the min rule: the largest bearish magnitude is taken off
    assert rk.combine([fr("event", "bearish", edge=-0.002), fr("capital_return", "bearish", edge=-0.001)]) == (-0.002, False)
    assert rk.combine([fr("trend", passed=False)]) == (None, False) and rk.combine([]) == (None, False)


def blend_of(value):
    """A blend with this value (the parts are not needed here: rank_symbol reads `value`)."""
    return {"value": value, "score": round(50 + 50 * value)}


def test_a_bearish_edge_has_the_confidence_of_the_bearish_side_of_the_blend():
    worse = scored(families=[fr("event", "bearish", edge=-0.008)], blend=blend_of(-0.6))
    better = scored(families=[fr("event", "bearish", edge=-0.008)], blend=blend_of(0.6))
    assert worse.confidence == pytest.approx(0.8) and better.confidence == pytest.approx(0.2)       # a bearish blend is the confidence of a bearish composite
    assert not hasattr(worse, "excess") and not hasattr(worse, "cost_pct") and not hasattr(worse, "scale_factor")


def test_an_edge_of_zero_is_neutral_and_says_so_and_any_edge_above_zero_is_bullish():
    s = scored(families=[fr("trend", edge=0.0)])
    assert s.stance == "neutral" and s.stance_reason == "no edge" and s.score == 50
    assert scored(families=[fr("trend", edge=0.0001)]).stance == "bullish"


def test_a_bearish_composite_is_bearish_and_uses_the_bearish_agreement():
    s = scored(families=[fr("event", "bearish", edge=-0.008)], votes=[("bearish", 1.0)] * 2)
    assert s.stance == "bearish" and s.composite == -0.008 and s.agreement == pytest.approx(1 - 1 / 4) and s.score == 50 and scored(families=[fr("event", "bearish", edge=-0.008)], blend=blend_of(-0.4)).score == 30   # two bearish news votes: bullish agreement (0+1)/(0+2+2) = 0.25, so the bearish one is 0.75


# ---------------------------------------------------------------- invalidation, cooldown, holdings

RULES = [{"kind": "price_below", "value": 90.0}, {"kind": "days_elapsed", "value": 63}, {"kind": "event_class", "value": ["guidance_cut"]}, {"kind": "metric_crossed", "value": "below_sma_200"}]


def record(action="open", family="trend", stop_date="2026-12-01", rules=RULES, rid="abc123"):
    return {"symbol": "AAA", "identity": {"record_id": rid}, "action": {"action": action}, "trade": {"family": family, "time_stop_date": stop_date, "invalidation": rules}}


def test_invalidation_rules_fire_on_the_price_a_bearish_event_and_the_family_condition():
    assert rk.invalidation_fired(RULES, healthy(last_close=100.0), []) == []
    assert rk.invalidation_fired(RULES, healthy(last_close=89.99), []) == ["price_below"]
    assert rk.invalidation_fired(RULES, healthy(last_close=100.0), [{"class": "guidance_cut"}]) == ["event_class"]
    assert rk.invalidation_fired(RULES, healthy(last_close=100.0, above_sma_200=False), []) == ["metric_crossed:below_sma_200"]
    assert rk.invalidation_fired([{"kind": "metric_crossed", "value": "rsi_14_above_70"}], healthy(rsi_14=71.0), []) == ["metric_crossed:rsi_14_above_70"]
    assert rk.invalidation_fired([{"kind": "metric_crossed", "value": "dilution_current"}], healthy(), []) == []          # an older record's issuance rule is not evaluated


def test_the_cooldown_holds_until_the_time_stop_and_ends_when_an_invalidation_rule_fires():
    s = scored(families=[fr()])
    assert rk.cooldown_of(s, "trend", [record()], "2026-10-23") == "abc123"
    assert rk.cooldown_of(s, "trend", [record(stop_date="2026-10-22")], "2026-10-23") is None            # the time stop passed
    assert rk.cooldown_of(s, "trend", [record(stop_date="2026-10-23")], "2026-10-23") == "abc123"        # the day itself still counts
    assert rk.cooldown_of(s, "reversion", [record()], "2026-10-23") is None                              # another family
    assert rk.cooldown_of(s, "trend", [record(action="hold")], "2026-10-23") is None
    broken = scored(families=[fr()], metrics=healthy(last_close=80.0))
    assert rk.cooldown_of(broken, "trend", [record()], "2026-10-23") is None                             # its stop was hit
    assert rk.cooldown_of(s, "trend", [], "2026-10-23") is None


def hold(s, records=(), floor=None):
    return rk.holding_action(s, list(records), floor)


def test_the_holdings_pass_reduces_on_a_red_flag_a_bearish_event_an_invalidation_or_a_family_that_stopped_passing():
    assert hold(scored(metrics=healthy(cfo_negative_with_profit=True), held=True))[0] == "reduce"
    action, reason = hold(scored(events=[{"class": "guidance_cut", "direction": "bearish", "age_bars": 3}], held=True))
    assert action == "reduce" and "guidance_cut" in reason
    assert hold(scored(events=[{"class": "guidance_cut", "direction": "bearish", "age_bars": 11}], held=True))[0] == "hold"
    action, reason = hold(scored(metrics=healthy(last_close=80.0), held=True), [record()])
    assert action == "reduce" and "abc123" in reason and "price_below" in reason
    stopped = scored(families=[fr("trend", passed=False)], held=True)
    action, reason = hold(stopped, [record()])
    assert action == "reduce" and "the family that recommended it (trend) no longer passes" in reason
    bearish = scored(families=[fr("trend", "bearish", edge=-0.002)], held=True)
    assert hold(bearish, [record()])[0] == "reduce"


def test_the_holdings_pass_adds_only_a_bullish_name_that_would_have_made_the_list():
    """Entry 80: the pass used to ask also that the plan was not "oversized" and that the position after adding stayed within 20% of the equity. Both needed a size and the user's money, which the Advisor
    does not use; whether the user's limits leave room to add is the Account Manager's (`sizing.size_purchases` counts the shares held against the position limit). What is left is the stance and the blend."""
    good = scored(families=[fr(edge=0.01)], held=True)
    assert hold(good)[0] == "add" and hold(good, floor=rk.rank_value(good))[0] == "add" and hold(good, floor=rk.rank_value(good) + 0.001)[0] == "hold"
    assert "position after" not in hold(good)[1] and "below this run's new ideas" in hold(good, floor=rk.rank_value(good) + 0.001)[1]
    assert not {"oversized_pct", "add_max_position_pct"} & set(type(CONFIG.ranking).model_fields)                        # the two size thresholds left the settings with the rule that read them
    assert hold(scored(families=[fr(passed=False)], held=True))[0] == "hold"
    assert hold(scored(families=[fr("trend", passed=True, edge=0.01)], held=True), [record()])[0] == "add"      # the family that recommended it still passes


# ---------------------------------------------------------------- the sleeves

def pick(candidates, case="funds", regime="trend"):
    chosen, why = rk.select(candidates, case, regime, CONFIG)
    return [(s.symbol, f.family, extra["diversity_adjustment"], extra["displaced_by"]) for s, f, extra in chosen], why


def test_an_event_name_does_not_open_when_a_capital_return_name_qualifies_and_is_the_fallback_when_none_does():
    event_only = scored("EV", [fr("event", edge=0.009, horizon_bars=40)])
    both = [event_only, scored("CR", [fr("capital_return", edge=0.004)])]
    names, why = pick(both)
    assert names == [("CR", "capital_return", False, None)] and why["EV"] == "only an event family supports it, and other names qualify"
    names, why = pick([event_only])
    assert names == [("EV", "event", False, None)]


def test_a_bearish_event_vote_vetoes_an_open_in_funds_but_not_in_other_sleeves():
    veto = scored("VT", [fr("capital_return", edge=0.006), fr("event", "bearish", edge=-0.001)])
    assert pick([veto])[0] == [] and pick([veto])[1]["VT"] == "vetoed by a bearish event vote"
    assert pick([veto], case="theme")[0] == []                                                          # theme needs the theme family: reported below
    assert pick([scored("VT", [fr("theme", edge=0.005), fr("event", "bearish", edge=-0.001)])], case="theme")[0][0][:2] == ("VT", "theme")


def test_names_that_are_not_bullish_or_are_gated_are_left_out_with_the_reason():
    cands = [scored("NEU", [fr("trend", edge=0.0)]), scored("GAT", [fr("capital_return", edge=0.006)], metrics=healthy(cfo_negative_with_profit=True)), scored("NON", [fr("trend", passed=False)])]
    names, why = pick(cands)
    assert names == [] and why["NEU"] == "stance neutral (no edge)" and why["GAT"] == "gate red_flag failed" and why["NON"] == "stance neutral (no family passed)"


def test_the_theme_sleeve_needs_the_theme_family_and_takes_up_to_five():
    leader = scored("LD", [fr("capital_return", edge=0.006)], theme_role="leader")
    names, why = pick([leader] + [scored(f"T{i}", [fr("theme", edge=0.004 + i * 0.0005, horizon_bars=42)]) for i in range(6)], case="theme")
    assert [n[0] for n in names] == ["T5", "T4", "T3", "T2", "T1"] and why["LD"] == "the theme family did not pass (theme_role: leader)" and why["T0"] == "rank 6 is beyond the top 5"


def test_diversity_takes_a_different_sector_when_its_blend_is_within_80_percent_and_records_it():
    cands = [scored("A1", [fr("capital_return", edge=0.010)], sector="XLK", blend=blend_of(0.6)), scored("A2", [fr("capital_return", edge=0.009)], sector="XLK", blend=blend_of(0.5)),
             scored("B1", [fr("capital_return", edge=0.0085)], sector="XLP", blend=blend_of(0.45)), scored("C1", [fr("capital_return", edge=0.003)], sector="XLE", blend=blend_of(0.1))]
    names, why = pick(cands)
    # A1 first; the next in rank is A2 (same sector): B1's blend 0.45 is >= 80% of A2's 0.5 (0.4), so B1 is taken and A2 is displaced
    assert names[0] == ("A1", "capital_return", False, None) and names[1] == ("B1", "capital_return", True, "A2") and names[2][0] == "A2"
    far = [scored("A1", [fr("capital_return", edge=0.010)], sector="XLK", blend=blend_of(0.6)), scored("A2", [fr("capital_return", edge=0.009)], sector="XLK", blend=blend_of(0.5)),
           scored("B1", [fr("capital_return", edge=0.0045)], sector="XLP", blend=blend_of(0.3))]
    assert [n[0] for n in pick(far)[0]] == ["A1", "A2", "B1"]                                            # B1's 0.3 is under 80% of A2's 0.5: no change
    unknown = [scored("A1", [fr("capital_return", edge=0.010)]), scored("A2", [fr("capital_return", edge=0.009)])]
    assert all(n[2] is False for n in pick(unknown)[0])                                                 # no sector known: no diversity rule (correlation is not built)


def test_a_bearish_company_ranks_below_one_with_no_signal_and_the_blend_orders_the_ones_with_no_signal():
    none_a = scored(symbol="AAA", families=[fr("trend", passed=False)])
    none_b = scored(symbol="BBB", families=[fr("trend", passed=False)], blend=blend_of(0.3))                                # no signal, but good news, prices and numbers
    bearish = scored(symbol="CCC", families=[fr("event", "bearish", edge=-0.008)])
    order = sorted([bearish, none_a, none_b], key=rk.sort_key)
    assert [s.symbol for s in order] == ["BBB", "AAA", "CCC"] and none_b.score > none_a.score == bearish.score
    tie = sorted([scored(symbol="ZZZ", families=[fr("trend", passed=False)]), scored(symbol="YYY", families=[fr("trend", passed=False)])], key=rk.sort_key)
    assert [s.symbol for s in tie] == ["YYY", "ZZZ"]                                                                        # a full tie ends on the symbol: the same order every run
