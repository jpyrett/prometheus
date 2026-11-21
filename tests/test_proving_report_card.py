"""The report card of the Critic's rules (prometheus/proving/report_card.py; Critic v2 rule C1, design/critic_v2_evidence_spec.md section 6) over the REAL runs and the REAL research of `tests/fixtures/proving_real/` (the oil test of six months, made by
Critic version 1): free, no model, no network. The numbers are checked against a plain join of the run's outcomes and the reviews of its research written here, with nothing of `report_card` in it."""
import glob
import json
import statistics
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from prometheus.application.web.app import create_app
from prometheus.application.web.proving_run import Proving
from prometheus.application.web.replay import Replay, load_capture
from prometheus.proving import report_card as rc
from prometheus.proving import stats
from prometheus.proving.record import ProvingRun
from prometheus.proving.store import ProvingStore

HERE = Path(__file__).parent
REAL = HERE / "fixtures" / "proving_real"
FAST = ProvingRun.model_validate_json((REAL / "oil_weekly_6m.json").read_text(encoding="utf-8"))      # the oil test of six months, asked as the live desk asks (the faithful speed); its research holds a denial for the user's risk profile
FOLDERS = sorted(REAL.glob("*_cache"))


def reviews_of_the_research() -> dict[str, dict]:
    """{record_id: its review} from every real research file beside the real runs (a plain read, written for this test)."""
    found = {}
    for path in glob.glob(str(REAL / "*_cache" / "*.json")):
        order = json.loads(Path(path).read_text(encoding="utf-8")).get("order")
        for review in (order or {}).get("reviews", []):
            found[review["record_id"]] = review
    return found


def by_hand(run: ProvingRun):
    """The run's reviewed outcomes joined to their reviews: [(outcome, review)] with an excess return, the same without the ones denied for the user's risk profile, and all of them with or without an excess return."""
    reviews = reviews_of_the_research()
    seen, everyone = set(), []
    for o in run.outcomes:
        if o.verdict in ("approved", "denied") and o.record_id not in seen and o.record_id in reviews:
            seen.add(o.record_id)
            everyone.append((o, reviews[o.record_id]))
    rows = [(o, r) for o, r in everyone if o.excess_return_pct is not None and o.result in ("won", "lost", "open")]
    return rows, [(o, r) for o, r in rows if "risk_above_profile" not in [b["code"] for b in r["blockers"]]], everyone


def mean(values):
    values = list(values)
    return statistics.fmean(values) if values else None


def test_the_verdict_row_is_the_denied_against_the_approved_without_the_suitability_denials():
    card = rc.build([FAST], folders=FOLDERS)
    rows, kept, _ = by_hand(FAST)
    assert card["config_versions"] == ["1"] and card["themes"] == ["oil"]
    v = card["by_config_version"]["1"]["verdict"]
    denied = [o.excess_return_pct for o, _ in kept if o.verdict == "denied"]
    approved = [o.excess_return_pct for o, _ in kept if o.verdict == "approved"]
    assert (v["hit"]["n"], v["rest"]["n"]) == (len(denied), len(approved)) and denied and approved
    assert v["hit"]["mean_excess"] == pytest.approx(mean(denied), abs=1e-6) and v["rest"]["mean_excess"] == pytest.approx(mean(approved), abs=1e-6)
    assert v["gap"] == pytest.approx(mean(denied) - mean(approved), abs=1e-6)
    expected = stats.gap_of(denied, approved)
    assert (v["gap"], v["se"], v["inside_chance"]) == (expected["gap"], expected["se"], expected["inside_chance"])
    assert v["by_theme"]["oil"]["n"] == len(denied) and v["sentence_code"] in ("denies_winners", "picks_losers", "no_clear_difference", "too_few")
    assert card["by_config_version"]["1"]["suitability_excluded"] == len(rows) - len(kept) > 0                       # the oil test has risk-profile denials, and they are out of the comparison


# ---------------------------------------------------------------- the route (the page's contract) and the offline evaluation

@pytest.fixture
def proving(tmp_path):
    return Proving(store=ProvingStore(tmp_path), fixtures=REAL, paid=False)


@pytest.fixture
def http(tmp_path, proving):
    replay = Replay(load_capture(HERE / "fixtures" / "session_real" / "bob_jones_oil.json"), speed=0)
    return TestClient(create_app(replay=replay, proving=proving))


def test_the_route_answers_with_the_card_of_the_recorded_runs_and_the_track_record(http, proving):
    card = http.get("/api/proving/critic-report").json()
    assert set(card) == {"min_sample", "chance_band", "limits", "sentences", "runs", "themes", "config_versions", "unmatched", "by_config_version"}
    assert card["min_sample"] == 20 and card["chance_band"] == 2 and set(card["sentences"]) == set(rc.SENTENCES) and "1" in card["by_config_version"]
    version = card["by_config_version"]["1"]
    assert set(version) == {"config_version", "reviewed", "population", "suitability_excluded", "verdict", "rows", "by_family", "as_v3"}      # Critic v3: a version before 3 says what the version-3 rules would have made of the same findings
    row = version["rows"][0]
    assert set(row) == {"code", "kind", "suitability", "fired", "hit", "rest", "gap", "se", "inside_chance", "sentence_code", "sentence", "by_theme"} and set(row["hit"]) == {"n", "closed", "won", "mean_excess"}
    assert {r["kind"] for r in card["runs"]} >= {"time_machine"} and any(r["kind"] == "track" for r in card["runs"])
    assert card == json.loads(json.dumps(proving.critic_report())) and http.get("/api/proving/critic-report").json() == card      # the route is the service, and the same answer twice


def test_the_offline_dataset_carries_the_finding_codes_and_critic_rule_value_scores_each_code():
    from prometheus.evals import datasets
    from prometheus.evals import evaluators as ev
    examples = datasets.outcome_examples()
    coded = [e for e in examples if e["inputs"].get("findings")]
    assert coded and all(set(e["inputs"]["findings"]) == {"config_version", "blockers", "warnings", "human"} for e in coded)
    rows = ev.evaluate_locally(examples, lambda inputs: next(e["outputs"] for e in examples if e["key"] == inputs["example_key"]), ev.OUTCOME_EVALUATORS)
    results = ev.flatten(ev.critic_rule_value(rows))
    assert results and all(r["key"].startswith("critic_rule_value:v") for r in results)
    one = results[0]
    version, code = one["key"].split(":")[1][1:], one["key"].split(":")[2]
    mine = [(set(e["inputs"]["findings"]["blockers"] + e["inputs"]["findings"]["warnings"] + e["inputs"]["findings"]["human"]), e["outputs"]["excess_return_pct"]) for e in coded
            if e["inputs"]["findings"]["config_version"] == version and e["inputs"]["verdict"] in ("approved", "denied") and isinstance(e["outputs"].get("excess_return_pct"), float)
            and "risk_above_profile" not in e["inputs"]["findings"]["blockers"]]
    assert one["score"] == stats.gap_of([x for c, x in mine if code in c], [x for c, x in mine if code not in c])["gap"]
    assert ev.critic_rule_value([{"example": {"inputs": {"verdict": "approved"}, "outputs": {}}, "outputs": {}}]) == ev.NOT_APPLICABLE


# ---------------------------------------------------------------- Critic version 3: the groups (design/critic_v3_two_reasons_spec.md section 5)

NEWS = {"fresh_negative_event", "fresh_negative_news", "event_not_supported", "analysts_bearish"}
PERFORMANCE = {"trend_not_confirmed", "dead_cat_bounce", "falling_knife", "price_below_zone", "chart_bearish", "poor_performance"}
PRICE = {"overpriced", "valuation_extreme", "price_left_zone"}


def hand_groups(entry):
    codes = {*entry.blockers, *entry.warnings, *entry.human}
    return [name for name, signs in (("news", NEWS), ("performance", PERFORMANCE), ("price", PRICE)) if codes & signs]


def real_entries():
    entries, _ = rc.entries_of([FAST], rc.ReviewIndex([FAST], FOLDERS))
    return entries


def test_the_groups_of_a_version_3_card_count_each_group_the_one_group_approvals_and_the_two_group_denials():
    """The real entries of the oil test given the groups their findings' codes make (by the hand-written table above) and labelled version 3, as the engine would write them for a run of the new Critic: every group row, the two-group verdict
    and the one-group row are the plain counts and means of those entries."""
    entries = real_entries()
    later = [rc.Entry(**{**e.__dict__, "version": "3", "record_id": e.record_id + "-v3", "groups": hand_groups(e)}) for e in entries]
    card = rc._version_card("3", later, 20, ["oil"])
    assert "as_v3" not in card and set(card["groups"]) == {"rows", "two", "one"} and [r["group"] for r in card["groups"]["rows"]] == ["news", "performance", "price"]
    population = [e for e in later if not e.suitability]
    for row in card["groups"]["rows"]:
        hit = [e for e in population if row["group"] in e.groups]
        clean = [e for e in population if row["group"] not in e.groups]
        assert row["fired"] == len(hit) and row["denied"] == sum(1 for e in hit if e.verdict == "denied") and row["kind"] == "group"
        assert (row["hit"]["n"], row["rest"]["n"]) == (len([e for e in hit if e.excess is not None]), len([e for e in clean if e.excess is not None]))
        assert row["hit"]["mean_excess"] == pytest.approx(mean(e.excess for e in hit if e.excess is not None), abs=1e-6)
    many, one = [e for e in population if len(e.groups) >= 2], [e for e in population if len(e.groups) == 1]
    assert card["groups"]["two"]["fired"] == len(many) and card["groups"]["one"]["fired"] == len(one) and one and all(row["sentence_code"] in rc.SENTENCES for row in [*card["groups"]["rows"], card["groups"]["two"], card["groups"]["one"]])
    text = rc.report({"min_sample": 20, "chance_band": 2.0, "limits": [], "runs": [], "themes": ["oil"], "unmatched": 0, "config_versions": ["3"], "by_config_version": {"3": card}})
    assert "CRITIC VERSION 3" in text and "the groups (a group is bad when a sign of it fired" in text and "two or more bad groups" in text and "one bad group" in text and "WHAT THE VERSION-3" not in text


