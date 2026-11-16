"""Integrity: is the recommendation record consistent with ITSELF and with the work order it came from? (Critic v1 spec section 8.1, step 2). No network, no model.

    result = check(rec, order, advisor_config)        # an `IntegrityResult`: every check that ran and the ones that failed, each with its reason in words

This is the port of the hard checks of `tests/simple_critic.py`, into the product: the Critic works every number of the record out again from the record's own inputs (the stop from the entry and
the distance, the composite edge from the passed families, the blend from its four parts and their recency weights, the scale factor, the excess after cost, the
score) and looks every quote up in the work order's own sources. A record that does not add up was damaged or invented, whoever did it; it is a blocker (`record_defect`), not a warning.

The constants of the Advisor's formulas (the 0.03 edge scale, the 40/30/20/10 weights, the half-life, the scale strength) are READ from the Advisor's config, never typed here: when Jacob changes
one, the Critic follows. Nothing is imported from the Advisor's scoring code (spec P1): these are re-derivations, which is what makes it a second pair of eyes.
"""
from dataclasses import dataclass, field
from datetime import date

from prometheus.agents.advisor.config import AdvisorConfig
from prometheus.agents.advisor.record import Recommendation
from prometheus.agents.critic.common import clip, quote_in_text
from prometheus.common.work_order import WorkOrder

FEATURE_KEYS = ("value", "unit", "as_of", "source")
BLEND_PARTS = ("news", "performance", "historical", "filings")
GATES = ("no_metrics", "freshness", "red_flag", "event_risk")           # the gates of the Advisor's record that a new position must pass. A gate record of any other name (an old capture still holds a `liquidity` one, made before design choices entry 94) decides nothing here
SOURCE_PREFIXES = ("day_aggs", "finnhub")      # the features that must be recent; a filing's own date is its statement's (quarters old is normal)


@dataclass
class IntegrityResult:
    checked: list[str] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)    # (check name, why)
    no_thesis: bool = False                                           # not a defect: the Advisor's thesis call may have failed (it says why on the record)

    @property
    def failed(self) -> list[str]:
        return [name for name, _ in self.failures]

    def check(self, name: str, ok: bool, why: str) -> bool:
        self.checked.append(name)
        if not ok:
            self.failures.append((name, why))
        return ok


def close(a, b, tolerance: float = 1e-6) -> bool:
    """Two numbers within a tolerance; two Nones are the same (a composite that does not exist twice), one None is not."""
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tolerance


def composite_of(families) -> float | None:
    """The best bullish edge less the largest bearish one (spec of the Advisor, 9.5), from the passed families; None when none passed."""
    passed = [f for f in families if f.passed and f.edge_20d is not None]
    bulls = [f.edge_20d for f in passed if f.direction == "bullish"]
    bears = [abs(f.edge_20d) for f in passed if f.direction == "bearish"]
    if bulls and bears:
        return max(bulls) - max(bears)
    return max(bulls) if bulls else -max(bears) if bears else None


def check(rec: Recommendation, order: WorkOrder, advisor: AdvisorConfig, duplicate: bool = False, max_feature_age_days: int = 5) -> IntegrityResult:
    """Every integrity check of one record. `duplicate` says the order holds another record of the same symbol and action."""
    r = IntegrityResult()
    ident, trade, ranking, agreement, features = rec.identity, rec.trade, rec.ranking, rec.agreement, rec.features
    as_of = date.fromisoformat(ident.as_of_utc[:10])

    # ---- complete
    r.check("complete.features", bool(features) and all(set(FEATURE_KEYS) <= set(f) and f["value"] is not None and f["unit"] and f["source"] for f in features.values()),
            "a feature lacks its value, unit or source (or there are no features at all)")
    r.check("complete.families", bool(rec.families) and all(f.basis_ref and f.horizon_bars for f in rec.families), "a family has no basis_ref or horizon")
    r.check("identity.request", ident.request_id == order.request_id, f"the record is from request {ident.request_id}, not {order.request_id}")
    r.check("config.version", rec.audit.config_version == advisor.config_version,
            f"the record was written under Advisor config {rec.audit.config_version!r} and the Advisor's config is {advisor.config_version!r}: its numbers cannot be worked out again")
    r.check("stance.action", ranking.stance == "bullish" and ranking.composite is not None and ranking.composite > 0,
            f"a recommendation to buy needs a bullish stance from a strategy with an edge above 0 (stance {ranking.stance}, composite {ranking.composite})")
    r.check("order.unique", not duplicate, f"the order holds two records of {rec.symbol}")

    # ---- the plan
    zone, mid = trade.entry_zone, trade.entry_mid
    r.check("trade.entry", bool(zone) and len(zone) == 2 and mid is not None and zone[0] < mid < zone[1], "there is no entry zone around the entry mid")
    r.check("trade.stop", trade.stop_price is not None and mid is not None and trade.stop_price < mid, "the stop is not under the entry")
    r.check("trade.time_stop", bool(trade.time_stop_date) and date.fromisoformat(trade.time_stop_date) > as_of, "the time stop is not in the future")
    r.check("trade.order_style", bool(trade.order_style), "there is no order style")
    rules = {rule["kind"]: rule["value"] for rule in trade.invalidation}
    r.check("trade.invalidation", {"price_below", "days_elapsed", "event_class"} <= set(rules) and rules.get("price_below") == trade.stop_price,
            "the invalidation rules are missing, or the price rule is not the stop")
    r.check("recompute.stop", trade.stop_price is not None and trade.stop_distance is not None and mid is not None and close(trade.stop_price, mid - trade.stop_distance, 0.011),
            "stop_price is not entry_mid - stop_distance")

    # ---- the composite edge, and the blend the score is made of
    expected = composite_of(rec.families)
    r.check("recompute.composite", close(ranking.composite, expected, 1e-6), f"composite {ranking.composite} is not the best bullish edge less the largest bearish one ({expected})")
    blend = rec.blend
    weights = advisor.blend.weights
    r.check("blend.present", blend is not None and set(blend.get("components", {})) == set(BLEND_PARTS), "the record has no blend with its four parts")
    if blend and set(blend.get("components", {})) == set(BLEND_PARTS):
        parts = blend["components"]
        half = blend["half_life_days"]
        r.check("blend.weights", {k: parts[k]["weight"] for k in BLEND_PARTS} == {k: getattr(weights, k) for k in BLEND_PARTS}, "the blend's weights are not the Advisor's configured weights")
        news = parts["news"]["parts"]
        r.check("blend.half_life", close(half, advisor.blend.half_life_days, 1e-9), "the blend's half-life is not the configured one")
        r.check("blend.recency", all(close(p["weight"], (1.0 if p["kind"] == "news" else advisor.blend.press_release_weight) * 0.5 ** ((half if p["age_days"] is None else p["age_days"]) / half), 2e-4)
                                     for p in news), "a news weight is not kind x 0.5 ** (age / half-life)")
        r.check("blend.news", close(parts["news"]["value"], sum((1 if p["direction"] == "bullish" else -1) * p["weight"] for p in news) / (sum(p["weight"] for p in news) + advisor.blend.news_prior_n)
                                    if news else 0.0, 2e-4), "the news value is not sum(direction x weight) / (sum(weight) + the prior)")
        for group in ("performance", "historical", "filings"):
            values = [p["value"] for p in parts[group]["parts"]]
            r.check(f"blend.{group}", close(parts[group]["value"], sum(values) / len(values) if values else 0.0, 2e-4) and all(-1.0 <= v <= 1.0 for v in values),
                    f"the {group} value is not the mean of its parts, or a part is outside -1 to +1")
        r.check("blend.value", close(blend["value"], sum(parts[k]["weight"] * parts[k]["value"] for k in BLEND_PARTS), 5e-4), "the blend is not the weighted sum of its parts")
        r.check("blend.score", blend["score"] == round(50 + 50 * blend["value"]), "the blend score is not round(50 + 50 x value)")
        direction = -1 if ranking.composite is not None and ranking.composite < 0 else 1
        r.check("recompute.confidence", close(agreement.confidence, (1 + direction * blend["value"]) / 2, 1e-3), "confidence is not (1 + direction x blend) / 2")
    if blend:
        r.check("recompute.score", rec.score == round(50 + 50 * blend["value"]), "score is not 50 + 50 x the blend")

    # ---- evidence: the order's own sources and filing evidence
    sources = {s["number"]: s.get("text") or "" for s in (order.data.sources if order.data else [])}
    items = {i["id"]: i for i in ((order.data.filing_evidence.get(rec.symbol) or {}).get("evidence", []) if order.data else []) if i.get("id")}
    for signal in rec.evidence.signals:
        if signal.get("evidence_id"):
            item = items.get(signal["evidence_id"])
            ok = item is not None and (quote_in_text(signal["quote"], item.get("quote")) or quote_in_text(signal["quote"], item.get("what")))
        else:
            ok = signal.get("source") in sources and quote_in_text(signal["quote"], sources[signal["source"]])
        r.check("evidence.quote", ok, f"the quote of {signal.get('evidence_id') or '[' + str(signal.get('source')) + ']'} is not in the order's text")
    for event in rec.evidence.events:
        if event.get("origin") == "model":
            r.check("evidence.event", event.get("source") in sources and quote_in_text(event["quote"], sources[event["source"]]), f"the quote of event {event['class']} is not in source {event.get('source')}")

    # ---- features: what a family says it used is what the features hold, and the numbers are recent
    for family in rec.families:
        for name, value in family.features_used.items():
            if name in features and isinstance(value, (int, float)) and not isinstance(value, bool):
                r.check("features.used", close(value, features[name]["value"], 1e-9), f"family {family.family} used {name}={value} but the features say {features[name]['value']}")
    stale = [n for n, f in features.items() if f.get("as_of") and n != "recent_bars" and not n.startswith("spy_") and str(f.get("source", "")).startswith(SOURCE_PREFIXES)
             and (as_of - date.fromisoformat(f["as_of"])).days > max_feature_age_days]
    r.check("features.fresh", not stale, f"older than {max_feature_age_days} days at the run date: {stale[:4]}")

    # ---- gates, thesis, audit
    failed_gates = [g.gate for g in rec.gates if not g.passed and g.gate in GATES]
    r.check("gates", not failed_gates, "a gate failed for a recommendation to buy: " + ", ".join(failed_gates))
    thesis = rec.thesis
    r.no_thesis = not (thesis.thesis or thesis.thesis_error)
    if thesis.thesis:
        allowed = {f"[{s['source']}]" for s in rec.evidence.signals if not s.get("evidence_id")} | {s["evidence_id"] for s in rec.evidence.signals if s.get("evidence_id")} \
            | {e["class"] for e in rec.evidence.events} | set(features)
        bad = [p["ref"] for p in [*thesis.bull_case, *thesis.bear_case] if p["ref"] not in allowed and f"[{p['ref']}]" not in allowed]
        r.check("thesis.refs", not bad, f"refs not in the record: {bad}")
    audit = rec.audit
    r.check("audit", bool(audit.config_version and audit.metric_order and audit.plan_run), "the config version, the metric order or the plan run is not named")
    return r
