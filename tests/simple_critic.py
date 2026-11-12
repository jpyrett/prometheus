"""A very simple Critic, only to test that the Advisor's recommendations carry what a Critic needs. It is NOT the Critic agent (Jacob writes that one next).

    result = critique(record, order)        # a Critique: verdict "reject" | "review" | "approve", failures, warnings

It reads ONE recommendation record (a dict, as `order.recommendations` holds it) and the work order it came from, and it knows things the Advisor was never told, which is what a second
pair of eyes is for. It uses no model and imports nothing from the Advisor's modules except the record's own schema: if a check needs a number, it works it out again from the record.

HARD CHECKS (a failure is a defect of the record: `reject`)
  complete     every section of the schema is there and has what the Critic reads; every feature has value, unit, as_of and source
  trade        an `open` or `add` has an entry zone, a stop under the entry, a time stop in the future, at least one share, an order style and the invalidation rules, and the price rule
               says the stop price
  recompute    stop = entry mid - stop distance; position = shares x entry mid / equity; composite, the blend (news with its recency weights, performance, historical, filings), the confidence and the score (the blend) from their own inputs
  evidence     every verified quote is a substring of the source it names (the Critic reads the work order's sources itself); every filing item cited is in the order's evidence
  features     a number a family says it used is the number in `features`; no feature is older than 5 days at the run date
  gates        every gate passed for a new position
  thesis       there is one (or an error that says why); every ref it cites is in the record; it is a thesis, not a promise (Amendment 6 removed the "unmeasured" rule with the expected return)
  audit        the config version, the metric order and the plan run are named
WARNINGS (a judgement of the position, not a defect: `review`)
  the stock is far under its 52-week high or its 200-day mean, a 63-day drawdown worse than 30%, a stop more than 25% away, an oversized position, a thin edge (under 0.2% after cost), a
  position over 20% of equity, counter-evidence on file, inputs under half complete, funds not feasible.
"""
import json
import re
from dataclasses import dataclass, field
from datetime import date

REQUIRED_SECTIONS = ("identity", "action", "families", "quality", "evidence", "features", "agreement", "ranking", "trade", "risk", "portfolio_fit", "gates", "thesis", "audit")
FEATURE_KEYS = ("value", "unit", "as_of", "source")
MAX_FEATURE_AGE_DAYS = 5


@dataclass
class Critique:
    symbol: str
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return "reject" if self.failures else "review" if self.warnings else "approve"


def norm(text: str) -> str:
    return " ".join(re.sub(r"[\u2018\u2019]", "'", re.sub(r"[\u201c\u201d]", '"', text or "")).lower().split())


def close(a, b, tol=1e-6) -> bool:
    return a is not None and b is not None and abs(a - b) <= tol


def critique(record: dict, order) -> Critique:
    c = Critique(record.get("symbol", "?"))

    def check(name: str, ok: bool, why: str) -> None:
        c.checked.append(name)
        if not ok:
            c.failures.append(f"{name}: {why}")

    # ---- complete
    missing = [s for s in REQUIRED_SECTIONS if record.get(s) is None]
    check("complete", not missing, f"missing sections {missing}")
    if missing:
        return c
    features = record["features"]
    check("complete.features", bool(features) and all(set(FEATURE_KEYS) <= set(f) and f["value"] is not None and f["unit"] and f["source"] for f in features.values()), "a feature lacks value, unit or source")
    check("complete.families", bool(record["families"]) and all(f["basis_ref"] and f["horizon_bars"] for f in record["families"]), "a family has no basis_ref or horizon")
    action, trade, ranking, agreement = record["action"]["action"], record["trade"], record["ranking"], record["agreement"]
    as_of = date.fromisoformat(record["identity"]["as_of_utc"][:10])
    opening = action in ("open", "add")

    # ---- trade
    if opening:
        zone, mid = trade.get("entry_zone"), trade.get("entry_mid")
        check("trade.entry", bool(zone) and len(zone) == 2 and zone[0] < mid < zone[1], "no entry zone around the entry mid")
        check("trade.stop", trade.get("stop_price") is not None and mid is not None and trade["stop_price"] < mid, "the stop is not under the entry")
        check("trade.time_stop", bool(trade.get("time_stop_date")) and date.fromisoformat(trade["time_stop_date"]) > as_of, "the time stop is not in the future")
        check("trade.size", (trade.get("suggested_shares") or 0) >= 1 and bool(trade.get("order_style")), "no whole-share size or no order style")
        rules = {r["kind"]: r["value"] for r in trade.get("invalidation", [])}
        check("trade.invalidation", {"price_below", "days_elapsed", "event_class"} <= set(rules) and rules.get("price_below") == trade.get("stop_price"), "invalidation rules missing, or the price rule is not the stop")
        # ---- recompute
        check("recompute.stop", close(trade["stop_price"], trade["entry_mid"] - trade["stop_distance"], 0.011), "stop_price is not entry_mid - stop_distance")
        if trade.get("equity"):
            check("recompute.position", close(trade["position_pct"], trade["suggested_shares"] * trade["entry_mid"] / trade["equity"], 1e-3), "position_pct is not shares x entry / equity")
    passed = [f for f in record["families"] if f["passed"]]
    bulls = [f["edge_20d"] for f in passed if f["direction"] == "bullish"]
    bears = [abs(f["edge_20d"]) for f in passed if f["direction"] == "bearish"]
    expected = (max(bulls) - max(bears)) if bulls and bears else max(bulls) if bulls else -max(bears) if bears else None
    check("recompute.composite", close(ranking["composite"], expected, 1e-6), f"composite {ranking['composite']} is not the best bullish edge less the largest bearish one ({expected})")
    # ---- the blend (entry 55): 0.40 news (the newer the heavier) + 0.30 performance + 0.20 historical + 0.10 filings, recomputed from its own parts
    blend = record.get("blend")
    check("blend.present", blend is not None and set(blend.get("components", {})) == {"news", "performance", "historical", "filings"}, "the record has no blend with its four parts")
    if blend and blend.get("components"):
        parts = blend["components"]
        check("blend.weights", {k: parts[k]["weight"] for k in parts} == {"news": 0.40, "performance": 0.30, "historical": 0.20, "filings": 0.10}, "the weights are not 40/30/20/10")
        news = parts["news"]["parts"]
        weights_ok = all(close(p["weight"], (1.0 if p["kind"] == "news" else 0.5) * 0.5 ** ((blend["half_life_days"] if p["age_days"] is None else p["age_days"]) / blend["half_life_days"]), 2e-4) for p in news)
        check("blend.recency", weights_ok, "a news weight is not kind x 0.5 ** (age / half-life)")
        check("blend.news", close(parts["news"]["value"], sum((1 if p["direction"] == "bullish" else -1) * p["weight"] for p in news) / (sum(p["weight"] for p in news) + 2) if news else 0.0, 2e-4), "the news value is not sum(direction x weight) / (sum(weight) + 2)")
        for group in ("performance", "historical", "filings"):
            values = [p["value"] for p in parts[group]["parts"]]
            check(f"blend.{group}", close(parts[group]["value"], sum(values) / len(values) if values else 0.0, 2e-4) and all(-1.0 <= v <= 1.0 for v in values), f"the {group} value is not the mean of its parts, or a part is outside -1 to +1")
        check("blend.value", close(blend["value"], sum(parts[k]["weight"] * parts[k]["value"] for k in parts), 5e-4), "the blend is not the weighted sum of its parts")
        check("blend.score", blend["score"] == round(50 + 50 * blend["value"]), "the blend score is not round(50 + 50 x value)")
        direction = -1 if ranking["composite"] is not None and ranking["composite"] < 0 else 1
        check("recompute.confidence", close(agreement["confidence"], (1 + direction * blend["value"]) / 2, 1e-3), "confidence is not (1 + direction x blend) / 2")
    if blend:
        check("recompute.score", record["score"] == round(50 + 50 * blend["value"]), "score is not 50 + 50 x the blend")

    # ---- evidence: the Critic reads the order's own sources
    sources = {s["number"]: norm(s["text"]) for s in (order.data.sources if order.data else [])}
    items = {i["id"]: i for e in (order.data.filing_evidence.values() if order.data else []) for i in e.get("evidence", [])}
    for s in record["evidence"]["signals"]:
        if s.get("evidence_id"):
            ok = s["evidence_id"] in items and norm(s["quote"]) in norm((items[s["evidence_id"]].get("quote") or "") + " " + (items[s["evidence_id"]].get("what") or ""))
        else:
            ok = s.get("source") in sources and norm(s["quote"]) in sources[s["source"]]
        check("evidence.quote", ok, f"the quote of source {s.get('evidence_id') or s.get('source')} is not in the order's text")
    for e in record["evidence"]["events"]:
        if e.get("origin") == "model":
            check("evidence.event", e.get("source") in sources and norm(e["quote"]) in sources[e["source"]], f"the quote of event {e['class']} is not in source {e.get('source')}")

    # ---- features
    for f in record["families"]:
        for name, value in f["features_used"].items():
            if name in features and isinstance(value, (int, float)) and not isinstance(value, bool):
                check("features.used", close(value, features[name]["value"], 1e-9), f"family {f['family']} used {name}={value} but features say {features[name]['value']}")
    stale = [n for n, f in features.items() if f.get("as_of") and n != "recent_bars" and not n.startswith("spy_") and (as_of - date.fromisoformat(f["as_of"])).days > MAX_FEATURE_AGE_DAYS and f["source"].startswith(("day_aggs", "finnhub"))]
    check("features.fresh", not stale, f"older than {MAX_FEATURE_AGE_DAYS} days at the run date: {stale[:4]}")

    # ---- gates, thesis, audit
    if opening:
        check("gates", all(g["passed"] for g in record["gates"]), "a gate failed for a new position: " + ", ".join(g["gate"] for g in record["gates"] if not g["passed"]))
    thesis = record["thesis"]
    check("thesis.present", bool(thesis.get("thesis")) or bool(thesis.get("thesis_error")), "no thesis and no reason")
    if thesis.get("thesis"):
        allowed = {f"[{s['source']}]" for s in record["evidence"]["signals"] if not s.get("evidence_id")} | {s["evidence_id"] for s in record["evidence"]["signals"] if s.get("evidence_id")} \
            | {e["class"] for e in record["evidence"]["events"]} | set(features)
        bad = [p["ref"] for p in [*thesis["bull_case"], *thesis["bear_case"]] if p["ref"] not in allowed and f"[{p['ref']}]" not in allowed]
        check("thesis.refs", not bad, f"refs not in the record: {bad}")
    audit = record["audit"]
    check("audit", bool(audit.get("config_version") and audit.get("metric_order") and audit.get("plan_run")), "config version, metric order or plan run is not named")

    # ---- warnings: the Critic's own judgement
    def feature(name):
        return (features.get(name) or {}).get("value")
    if opening:
        if feature("pct_of_52w_high") is not None and feature("pct_of_52w_high") < 0.5:
            c.warnings.append(f"far under its 52-week high ({feature('pct_of_52w_high'):.0%} of it)")
        if feature("above_sma_200") is False:
            c.warnings.append("below its 200-day mean")
        if feature("max_drawdown_63d") is not None and feature("max_drawdown_63d") < -0.30:
            c.warnings.append(f"a {feature('max_drawdown_63d'):.0%} drawdown in 63 days")
        if trade.get("stop_distance") and trade.get("entry_mid") and trade["stop_distance"] / trade["entry_mid"] > 0.25:
            c.warnings.append(f"the stop is {trade['stop_distance'] / trade['entry_mid']:.0%} away")
        if trade.get("oversized"):
            c.warnings.append("oversized: " + (trade.get("size_note") or ""))
        if (trade.get("position_pct") or 0) > 0.20:
            c.warnings.append(f"the position is {trade['position_pct']:.0%} of equity")
        if trade.get("funds_feasible") is False:
            c.warnings.append("the funds are not feasible")
        if agreement.get("completeness") is not None and agreement["completeness"] < 0.5:
            c.warnings.append(f"the inputs are {agreement['completeness']:.0%} complete")
    if record["evidence"]["counter_evidence"]:
        c.warnings.append("counter-evidence on file: " + ", ".join(x["ref"] for x in record["evidence"]["counter_evidence"][:4]))
    return c


def critique_order(order) -> list[Critique]:
    """A Critique for every recommendation on the order."""
    return [critique(r, order) for r in order.recommendations]


def tampered(record: dict, **changes) -> dict:
    """A copy of a record with a nested change, for the tests that prove a check can fail: `tampered(r, **{"trade.stop_price": 1.0})`."""
    out = json.loads(json.dumps(record))
    for path, value in changes.items():
        node = out
        *parents, last = path.split(".")
        for key in parents:
            node = node[key]
        node[last] = value
    return out
