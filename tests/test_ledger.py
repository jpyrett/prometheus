"""Tests for prometheus.memory.ledger (Advisor v1 spec section 14, design choices entry 51): append-only, recent, and a corrupt line reported and not swallowed."""
import json
from datetime import date

from prometheus.memory.ledger import DEFAULT_PATH, Ledger


def rec(symbol="PG", family="capital_return", day="2026-10-23", rid="a1"):
    return {"symbol": symbol, "identity": {"record_id": rid, "as_of_utc": f"{day}T12:00:00Z"}, "trade": {"family": family}, "action": {"action": "open"}}


def test_the_folder_is_made_on_the_first_write_and_a_missing_file_is_an_empty_ledger(tmp_path):
    ledger = Ledger(tmp_path / "ledger" / "r.jsonl")
    assert ledger.read() == [] and ledger.recent("PG") == [] and ledger.problems == []
    ledger.append(rec())
    assert (tmp_path / "ledger" / "r.jsonl").exists() and ledger.read() == [rec()]


def test_it_only_appends_one_line_per_record_and_never_rewrites_a_line(tmp_path):
    ledger = Ledger(tmp_path / "r.jsonl")
    ledger.append(rec(rid="a1"))
    first = (tmp_path / "r.jsonl").read_text()
    ledger.append(rec(rid="a2"))
    text = (tmp_path / "r.jsonl").read_text()
    assert text.startswith(first) and len(text.splitlines()) == 2 and [json.loads(line)["identity"]["record_id"] for line in text.splitlines()] == ["a1", "a2"]


def test_recent_filters_by_symbol_family_and_date_and_keeps_the_order(tmp_path):
    ledger = Ledger(tmp_path / "r.jsonl")
    for r in (rec("PG", "capital_return", "2026-09-01", "a1"), rec("PG", "trend", "2026-10-01", "a2"), rec("MU", "trend", "2026-10-02", "a3"), rec("PG", "capital_return", "2026-10-20", "a4")):
        ledger.append(r)
    assert [r["identity"]["record_id"] for r in ledger.recent("PG")] == ["a1", "a2", "a4"]
    assert [r["identity"]["record_id"] for r in ledger.recent("PG", family="capital_return")] == ["a1", "a4"]
    assert [r["identity"]["record_id"] for r in ledger.recent("PG", since=date(2026, 10, 1))] == ["a2", "a4"]
    assert ledger.recent("PG", family="reversion") == [] and ledger.recent("SNDK") == []


def test_a_corrupt_line_is_reported_with_its_number_and_the_rest_is_still_read(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps(rec(rid="a1")) + "\n" + "{not json, with a secret=hunter2\n" + json.dumps([1, 2]) + "\n" + "\n" + json.dumps(rec(rid="a2")) + "\n")
    ledger = Ledger(path)
    records = ledger.read()
    assert [r["identity"]["record_id"] for r in records] == ["a1", "a2"]
    assert ledger.problems == ["r.jsonl line 2: not valid JSON", "r.jsonl line 3: not a JSON object"]
    assert "hunter2" not in " ".join(ledger.problems)                         # a line's content is never put in a message
    assert [r["identity"]["record_id"] for r in ledger.recent("PG")] == ["a1", "a2"] and len(ledger.problems) == 2   # reading again does not stack the problems


def test_dates_and_other_non_json_values_are_written_as_text(tmp_path):
    ledger = Ledger(tmp_path / "r.jsonl")
    ledger.append({"symbol": "PG", "when": date(2026, 10, 23)})
    assert ledger.read() == [{"symbol": "PG", "when": "2026-10-23"}]


def test_the_default_path_is_the_ledger_folder_and_not_the_saved_evidence():
    assert DEFAULT_PATH.parts[-3:] == ("data", "ledger", "recommendations.jsonl") and "processed" not in DEFAULT_PATH.parts and "historical" not in DEFAULT_PATH.parts
