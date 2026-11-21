"""Tests for prometheus.agents.advisor.ranking (Advisor v1 spec sections 9.5 to 9.8 and 12, phase 5): one vote per family, the min rule, the cost, the stance, the regime, the holdings
pass, the cooldown, the sleeves, diversity and a deterministic order. Numbers are literals that put a rule at its threshold; they are not market data."""
import pytest
from advisor_helpers import CONFIG, fr, healthy, metrics_of, scored

from prometheus.agents.advisor import ranking as rk


# ---------------------------------------------------------------- the regime


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


def blend_of(value):
    """A blend with this value (the parts are not needed here: rank_symbol reads `value`)."""
    return {"value": value, "score": round(50 + 50 * value)}


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
