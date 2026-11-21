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
from prometheus.proving.record import ProvingRun, critic_cell
from prometheus.proving.store import ProvingStore

HERE = Path(__file__).parent
REAL = HERE / "fixtures" / "proving_real"
FAST = ProvingRun.model_validate_json((REAL / "oil_weekly_6m.json").read_text(encoding="utf-8"))      # the oil test of six months, asked as the live desk asks (the faithful speed); its research holds a denial for the user's risk profile
FASTER = ProvingRun.model_validate_json((REAL / "oil_weekly_6m_fast.json").read_text(encoding="utf-8"))      # the same test asked fast: its own research, other recommendations
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


def test_every_finding_code_has_a_row_with_its_counts_and_its_gap_against_everything_else():
    card = rc.build([FAST], folders=FOLDERS)
    rows, kept, everyone = by_hand(FAST)
    table = {r["code"]: r for r in card["by_config_version"]["1"]["rows"]}
    wanted = {"blend_not_supported", "fresh_negative_news", "company_mismatch", "overpriced", "rich_price", "risk_above_profile"}
    assert wanted <= set(table)
    for code in ("blend_not_supported", "fresh_negative_news", "rich_price"):
        has = lambda review: code in [f["code"] for f in review["blockers"] + review["warnings"]] or code in review["requires_human_reasons"]
        hit, rest = [o.excess_return_pct for o, r in kept if has(r)], [o.excess_return_pct for o, r in kept if not has(r)]
        row = table[code]
        assert row["fired"] == sum(1 for o, r in everyone if has(r)) and (row["hit"]["n"], row["rest"]["n"]) == (len(hit), len(rest))
        assert row["hit"]["mean_excess"] == pytest.approx(mean(hit), abs=1e-6) and row["gap"] == pytest.approx(mean(hit) - mean(rest), abs=1e-6)
        assert row["hit"]["won"] == sum(1 for o, r in kept if has(r) and o.result == "won") and row["hit"]["closed"] == sum(1 for o, r in kept if has(r) and o.result in ("won", "lost"))
    assert table["blend_not_supported"]["kind"] == "blocker" and table["fresh_negative_news"]["kind"] == "human" and table["company_mismatch"]["kind"] == "human" and table["rich_price"]["kind"] == "warning"
    ordered = card["by_config_version"]["1"]["rows"]
    assert table["risk_above_profile"]["suitability"] and ordered[-1]["suitability"] and not any(r["suitability"] for r in ordered[:-1])      # the suitability rows come last


def test_a_suitability_rule_is_listed_with_its_counts_and_never_judged_by_returns():
    row = next(r for r in rc.build([FAST], folders=FOLDERS)["by_config_version"]["1"]["rows"] if r["code"] == "risk_above_profile")
    assert row["suitability"] and row["fired"] > 0 and (row["gap"], row["se"], row["inside_chance"]) == (None, None, None) and row["sentence_code"] == "suitability" and "never judged by returns" in row["sentence"]
    assert all(t["gap"] is None and t["sentence_code"] == "suitability" for t in row["by_theme"].values()) and set(rc.SUITABILITY) >= {"risk_above_profile"}


def test_a_recommendation_in_two_runs_counts_once_and_a_small_group_is_too_few_to_say():
    twice = rc.build([FAST, FAST], folders=FOLDERS)
    once = rc.build([FAST], folders=FOLDERS)
    assert twice["by_config_version"] == once["by_config_version"]
    small = [r for r in once["by_config_version"]["1"]["rows"] if not r["suitability"] and min(r["hit"]["n"], r["rest"]["n"]) < once["min_sample"]]
    assert small and all(r["sentence_code"] == "too_few" and "20 are needed" in r["sentence"] for r in small)
    three = rc.build([FAST, FASTER], folders=FOLDERS)                                                       # two real runs with their own research: more recommendations in every group
    big = [r for r in three["by_config_version"]["1"]["rows"] if not r["suitability"] and min(r["hit"]["n"], r["rest"]["n"]) >= three["min_sample"]]
    assert big and all(r["sentence_code"] in ("denies_winners", "picks_losers", "no_clear_difference") for r in big)
    assert all((r["sentence_code"] == "no_clear_difference") == bool(r["inside_chance"]) for r in big)
    assert all((r["sentence_code"] == "denies_winners") == (r["gap"] > 0 and not r["inside_chance"]) for r in big)


def test_versions_are_never_mixed_in_one_row_and_the_sentence_is_one_of_the_closed_set():
    entries, _ = rc.entries_of([FAST], rc.ReviewIndex([FAST], FOLDERS))
    assert {e.version for e in entries} == {"1"} and entries
    later = [rc.Entry(**{**e.__dict__, "version": "2", "record_id": e.record_id + "-v2", "blockers": [b for b in e.blockers if b != "blend_not_supported"]}) for e in entries]
    themes = ["oil"]
    one, two = rc._version_card("1", entries, 20, themes), rc._version_card("2", later, 20, themes)
    assert "blend_not_supported" in {r["code"] for r in one["rows"] if r["kind"] == "blocker"} and "blend_not_supported" not in {r["code"] for r in two["rows"] if r["kind"] == "blocker"}      # the same recommendations, one rule less
    assert one["reviewed"] == two["reviewed"] == len(entries)
    for row in [one["verdict"], *one["rows"], *two["rows"]]:
        assert row["sentence_code"] in rc.SENTENCES and row["sentence"] == rc.sentence_of(row["sentence_code"], row, ("what it fired on", "the others") if row["code"] != "denied" else ("the denied", "the approved"), 20)


def test_a_run_saved_with_the_codes_gives_the_card_a_run_saved_without_them_gives_from_the_research():
    """The way the engine writes an outcome now (`critic_cell` of the review: the codes and the version) and the way the card reads an older one (the review in the research cache) end in the same card."""
    reviews = reviews_of_the_research()
    saved = FAST.model_copy(deep=True)
    for o in saved.outcomes:
        if o.critic is not None and o.record_id in reviews:
            o.critic = critic_cell(reviews[o.record_id])
    bare = rc.build([FAST], folders=FOLDERS)
    cells = rc.build([saved], folders=[])               # no research folder at all: everything it needs is on the outcomes
    assert cells["by_config_version"] == bare["by_config_version"]
    cell = critic_cell(next(r for r in reviews.values() if r["blockers"] and r["warnings"]))
    review = next(r for r in reviews.values() if r["blockers"] and r["warnings"])
    assert cell.blockers == [b["code"] for b in review["blockers"]] and cell.warnings == [w["code"] for w in review["warnings"]] and cell.human == review["requires_human_reasons"] and cell.config_version == "1"


def test_the_text_of_the_card_has_the_verdict_each_rule_and_the_limits():
    text = rc.report(rc.build([FAST], folders=FOLDERS))
    assert text.startswith("CRITIC REPORT CARD  1 run(s)") and "CRITIC VERSION 1:" in text and "the verdict" in text and "blend_not_supported" in text and "limit: " in text and "approved minus denied" in text


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
    assert set(version) == {"config_version", "reviewed", "population", "suitability_excluded", "verdict", "rows", "by_family"}
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
