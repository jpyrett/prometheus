"""The thesis: the second model call, written from the FINISHED record and checked by code (Advisor v1 spec section 11.2, design choices entry 51).

    theses = write_theses(model, records, run)         # {symbol: Thesis}: the model's three sentences and bullets, or None and the reason

By the time the model is asked, the program has already decided everything: the strategy, the stop, the target, the time stop (every record is a recommendation to buy: the Advisor knows no holding, entry 105). It has decided no
amount: how much to buy is the Account Manager's (entry 80), so the record the model reads has no size and the numbers the check allows have none. The model's job is only to EXPLAIN that in words, so it is given the record as a
fixed-wording digest (`digest`, a template filled from the record, like `filing_signals.digest`) and may use nothing else.
`model_score` is not in the digest: the model does not see its own earlier number again.

Code then checks the reply (`check`) and a thesis that fails is not shown:

  1. at most 3 sentences;
  2. every `ref` of a bullet is one the record has: a verified signal's source number (`[4]`), a filing-evidence id (`E2`), an event class that fired, or a metric name in `features`;
  3. every NUMBER in the thesis and the bullets is a number the record has (a value of a metric, the ranking, the trade, a number inside a verified quote), compared after rounding to the
     precision the text shows (`4.2%` matches 0.0421, `$163.45` matches 163.4512, `1.2 billion` matches 1.2e9); ISO dates are checked as text;
  4. when the record lists counter-evidence or contradictions, `bear_case` is not empty and cites at least one of them;

A failed thesis gets ONE retry with the reasons listed; a second failure leaves `thesis: null` and `thesis_error` naming the reasons, and the record is still returned: a recommendation does not
disappear because the prose was not good enough. The two sentences about a number and a ref are the whole anti-hallucination story for the prose: the model cannot say a figure the program did not calculate.
"""
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from prometheus.agents.advisor.calls import ask_json
from prometheus.agents.advisor.record import Thesis
from prometheus.prompts.advisor_v1 import RETRY_PLAN, RETRY_THESIS, THESIS_SYSTEM, thesis_message

MAX_SENTENCES = 3
ALWAYS_ALLOWED = (20.0,)   # the 20-day window of a strategy's edge
NUMBER = re.compile(r"(?<![\w.])\$?(-?\d[\d,]*(?:\.\d+)?)\s?(%|trillion|billion|million)?", re.I)
DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
REFERENCE = re.compile(r"\[\d+\]|\bE\d+\b")
SCALE = {"trillion": 1e12, "billion": 1e9, "million": 1e6}
# Text under these keys is an identifier or a date (a random record id, a chunk id, an accession number, a URL), not something the thesis may quote: the digits in it are not "numbers of the record".
# (A run was flaky before this: a random record id with the digits 37 in it made "37" an allowed number.)
IDENTIFIER_KEYS = {"chunk_id", "article_id", "url", "record_id", "request_id", "as_of_utc", "metric_order", "published_utc", "bar_date", "as_of", "time_stop_date", "source", "accession", "model"}


def _pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def _num(value, digits=4) -> str:
    return "n/a" if value is None else (f"{value:,.{digits}f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value))


def refs_of(record: dict) -> dict[str, str]:
    """The refs a bullet may cite, each with what it points at: {ref: kind}. Signal sources as `[4]`, evidence ids, event classes, feature names."""
    refs: dict[str, str] = {}
    for signal in (record.get("evidence") or {}).get("signals", []):
        if signal.get("evidence_id"):
            refs[signal["evidence_id"]] = "evidence"
        elif signal.get("source") is not None:
            refs[f"[{signal['source']}]"] = "source"
    for event in (record.get("evidence") or {}).get("events", []):
        refs[event["class"]] = "event"
    for name in record.get("features") or {}:
        refs.setdefault(name, "metric")
    return refs


def counter_refs(record: dict) -> set[str]:
    """The refs of the counter-evidence and the contradictions: the ones a bear case must cite at least one of."""
    out = {c.get("ref") for c in (record.get("evidence") or {}).get("counter_evidence", [])}
    out |= {ref for pair in (record.get("evidence") or {}).get("contradictions", []) for ref in pair}
    return {r for r in out if r}


def digest(record: dict) -> str:
    """The record as fixed wording for the model (a template, no prose of a model): the decision, the numbers, the plan, what each family said, the evidence, and the refs it may cite."""
    ident, action, ranking, trade = record["identity"], record["action"], record["ranking"], record["trade"]
    lines = [f"RECORD {record['symbol']}" + (f" ({record['company']})" if record.get("company") else ""),
             f"Decision: buy ({action['reason']}). Case {ident['case']}, regime {ident['regime']}" + (f" ({ident['regime_note']})" if ident.get("regime_note") else "") + "."]
    lines.append(f"Score: {record['score']} of 100 (the blend of news, performance, historical metrics and filings), stance {ranking['stance']}. No return is promised: the record has no expected return.")
    if trade.get("entry_zone"):
        lines.append(f"Trade: strategy {trade['family'] or 'none'}; entry zone {trade['entry_zone'][0]} to {trade['entry_zone'][1]} ({trade['entry_basis']}); stop {_num(trade['stop_price'])}; target {_num(trade['target_price'])}; "
                     f"time stop {trade['time_stop_date'] or 'none'}; order {trade['order_style'] or 'none'}.")
    for f in record["families"]:
        if f["passed"]:
            lines.append(f"Family {f['family']}: passed, {f['direction']}, score {_num(f['score'])}, expected 20-day edge {_pct(f['edge_20d'])}, horizon {f['horizon_bars']} bars; used " + ", ".join(f"{k}={_num(v)}" for k, v in f["features_used"].items() if isinstance(v, (int, float, bool))) + ".")
        elif f["computable"]:
            lines.append(f"Family {f['family']}: did not pass: " + "; ".join(f["notes"]) + ".")
        else:
            lines.append(f"Family {f['family']}: not computable, missing " + ", ".join(f["missing"]) + ".")
    q = record["quality"]
    lines.append(f"Quality multiplier {_num(q['multiplier'])}" + (": " + "; ".join(f"{h['check']} {_num(h['value'])} x{h['factor']}" for h in q["haircuts"]) if q["haircuts"] else "") + ".")
    lines.append("Gates: " + "; ".join(f"{g['gate']} {'ok' if g['passed'] else 'FAILED'}" + (f" ({g['note']})" if g.get("note") else "") for g in record["gates"]) + ".")
    ev = record["evidence"]
    for s in ev["signals"]:
        lines.append(f"Evidence {s.get('evidence_id') or '[' + str(s['source']) + ']'} {s['direction']}: \"{s['quote']}\"")
    for e in ev["events"]:
        lines.append(f"Event {e['class']} ({e['direction']}, {e.get('age_bars')} bars ago): \"{e['quote']}\"")
    if ev["counter_evidence"]:
        lines.append("Counter-evidence: " + "; ".join(f"{c['ref']} {c['direction']}" for c in ev["counter_evidence"]))
    if ev["contradictions"]:
        lines.append("Contradictions: " + "; ".join(f"{a} against {b}" for a, b in ev["contradictions"]))
    risk = record["risk"]
    lines.append("Risk: " + ", ".join(f"{k}={_num(v)}" for k, v in risk.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None) + (f"; red flags {', '.join(risk['red_flags'])}" if risk["red_flags"] else "") + ".")
    lines.append("ALLOWED REFS: " + ", ".join(refs_of(record)))
    return "\n".join(lines)


def _numbers_in(value, out: set[float]) -> None:
    """Every number in a record: its numeric leaves, and the numbers written inside its text (a quote's '12%', a note's '3 x atr_14')."""
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        out.add(float(value))
    elif isinstance(value, str):
        for match in NUMBER.finditer(REFERENCE.sub(" ", DATE.sub(" ", value))):
            number, unit = float(match.group(1).replace(",", "")), (match.group(2) or "").lower()
            out.add(number * SCALE[unit] if unit in SCALE else number)
    elif isinstance(value, dict):
        for key, item in value.items():
            if not (isinstance(item, str) and key in IDENTIFIER_KEYS):
                _numbers_in(item, out)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _numbers_in(item, out)


def allowed_numbers(record: dict) -> set[float]:
    """The numbers the thesis may use: everything in the record except the thesis itself."""
    out = set(ALWAYS_ALLOWED)
    _numbers_in({k: v for k, v in record.items() if k != "thesis"}, out)
    return out


def unknown_numbers(text: str, allowed: set[float], record_text: str) -> list[str]:
    """The numbers in `text` that the record does not have, as written. A number matches when rounding an allowed value to the digits the text shows gives it (a percent is compared with the
    value x 100 as well); `1.2 billion` is compared at full size; an ISO date must appear in the record's text; `[4]` and `E2` are refs, not numbers."""
    bad = [d for d in DATE.findall(text) if d not in record_text]
    stripped = REFERENCE.sub(" ", DATE.sub(" ", text))
    for match in NUMBER.finditer(stripped):
        raw, unit = match.group(1), (match.group(2) or "").lower()
        shown = float(raw.replace(",", ""))
        digits = len(raw.split(".")[1]) if "." in raw else 0
        if unit in SCALE:
            candidates = [(v / SCALE[unit]) for v in allowed]
        elif unit == "%":
            candidates = [*allowed, *(v * 100 for v in allowed)]
        else:
            candidates = list(allowed)
        if not any(abs(round(c, digits) - shown) < 1e-9 for c in candidates):
            bad.append(match.group(0).strip())
    return bad


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s]


def check(item: dict, record: dict) -> list[str]:
    """The reasons a model's thesis for this record fails the checks (an empty list means it passes)."""
    reasons = []
    text = item.get("thesis")
    if not isinstance(text, str) or not text.strip():
        return ["the thesis is empty"]
    bull, bear = item.get("bull_case"), item.get("bear_case")
    if not isinstance(bull, list) or not isinstance(bear, list) or not all(isinstance(p, dict) and isinstance(p.get("text"), str) and isinstance(p.get("ref"), str) for p in [*bull, *bear]):
        return ["bull_case and bear_case must be lists of {text, ref}"]
    if len(sentences(text)) > MAX_SENTENCES:
        reasons.append(f"the thesis has {len(sentences(text))} sentences (at most {MAX_SENTENCES})")
    allowed_refs = refs_of(record)
    for point in [*bull, *bear]:
        ref = point["ref"].strip()
        if ref not in allowed_refs and f"[{ref}]" not in allowed_refs:
            reasons.append(f"ref {point['ref']!r} is not one of the record's allowed refs")
    allowed, record_text = allowed_numbers(record), json.dumps({k: v for k, v in record.items() if k != "thesis"}, default=str)
    for where, words in (("thesis", text), *((f"{side} point", p["text"]) for side, points in (("bull", bull), ("bear", bear)) for p in points)):
        unknown = unknown_numbers(words, allowed, record_text)
        if unknown:
            reasons.append(f"the {where} has numbers that are not in the record: {', '.join(unknown)}")
    counters = counter_refs(record)
    if counters and not any(p["ref"].strip().strip("[]") in {c.strip("[]") for c in counters} for p in bear):
        reasons.append("the record lists counter-evidence or contradictions: bear_case must cite at least one of them (" + ", ".join(sorted(counters)) + ")")
    return reasons


def write_theses(model, records: list[dict], run: dict) -> dict[str, Thesis]:
    """Ask the model for every record's thesis in one call, check each, retry the failures ONCE, and return {symbol: Thesis}. A failure is a Thesis with `thesis: None` and `thesis_error`.
    The call is recorded in `run` (an entry of `advisor_runs`)."""
    by_symbol = {r["symbol"]: r for r in records}
    results = {symbol: Thesis(thesis=None, thesis_error="the thesis was not written") for symbol in by_symbol}
    if not records:
        return results
    messages = [SystemMessage(content=THESIS_SYSTEM), HumanMessage(content=thesis_message([digest(r) for r in records]))]
    items = ask_json(model, messages, run, "theses", RETRY_PLAN)
    if items is None:
        return {symbol: Thesis(thesis=None, thesis_error=run["error"]) for symbol in by_symbol}
    failures: dict[str, list[str]] = {}

    def take(items: list) -> None:
        seen = set()
        for item in items:
            symbol = item.get("symbol") if isinstance(item, dict) else None
            if symbol not in by_symbol or symbol in seen:
                continue
            seen.add(symbol)
            reasons = check(item, by_symbol[symbol])
            if reasons:
                failures[symbol] = reasons
            else:
                failures.pop(symbol, None)
                results[symbol] = Thesis(thesis=item["thesis"].strip(), bull_case=item["bull_case"], bear_case=item["bear_case"])
        for symbol in by_symbol:
            if symbol not in seen and symbol not in {s for s, t in results.items() if t.thesis}:
                failures.setdefault(symbol, ["the model gave no thesis for it"])

    take(items)
    if failures:
        retry = [*messages, HumanMessage(content=RETRY_THESIS.format(reasons="\n".join(f"- {s}: {'; '.join(r)}" for s, r in failures.items())))]
        again = ask_json(model, retry, run, "theses", RETRY_PLAN)
        if again is not None:
            wanted = set(failures)
            take([i for i in again if isinstance(i, dict) and i.get("symbol") in wanted])
    for symbol, reasons in failures.items():
        results[symbol] = Thesis(thesis=None, thesis_error="; ".join(reasons))
    return results
