"""The claims of a recommendation: the quotes it stands on, each with the source it was taken from (Critic v1 spec sections 7 and 8.2).

    claims = claims_of(rec, order)            # every counted, verified signal and every model-origin event of the record, one `Claim` each

The Advisor's model quoted text from the stored sources, and the Advisor's CODE proved each quote is in the source it names. What the Critic adds is to read the source AGAIN, from the store, today,
and then to ask a model of another producer whether the quote is really good news for THIS company. This module only lists what there is to check; it decides nothing.

A signal and an event that quote the same words from the same source are ONE claim (it carries the event's class too, so that "the event family's event was not supported" can be said). A different
quote from the same source is another claim. Refs are unique on a record: `[4]` for a source, `E15` for a filing-evidence item, the event's class for an event (a second event of the same class
gets `#2`).
"""
from dataclasses import dataclass, replace

from prometheus.agents.advisor.advisor import normalize
from prometheus.agents.advisor.record import Recommendation
from prometheus.common.work_order import WorkOrder


@dataclass(frozen=True)
class Claim:
    ref: str                       # [4] | E15 | an event class
    kind: str                      # signal | evidence | event
    direction: str                 # bullish | bearish
    quote: str
    source: int | None = None      # the source number on the order, for a signal or an event of a stored source
    chunk_id: str | None = None    # the stored chunk to read again; None for a filing-evidence item
    evidence_id: str | None = None
    event_class: str | None = None


def claims_of(rec: Recommendation, order: WorkOrder) -> list[Claim]:
    """Every counted and verified signal of the record and every event the model gave (`origin == "model"`; the filing index's events have no quote and are checked by their form and date)."""
    sources = {s.get("number"): s for s in (order.data.sources if order.data else [])}
    found: dict[tuple, Claim] = {}
    for signal in rec.evidence.signals:
        if not (signal.get("verified") and signal.get("counted")):
            continue
        evidence_id = signal.get("evidence_id")
        claim = Claim(ref=evidence_id or f"[{signal.get('source')}]", kind="evidence" if evidence_id else "signal", direction=signal["direction"], quote=signal["quote"],
                      source=None if evidence_id else signal.get("source"), chunk_id=None if evidence_id else signal.get("chunk_id"), evidence_id=evidence_id)
        found[(evidence_id or claim.chunk_id, normalize(claim.quote))] = claim
    for event in rec.evidence.events:
        if event.get("origin") != "model":
            continue
        source = sources.get(event.get("source"))
        chunk_id = source.get("chunk_id") if source else None
        key = (chunk_id, normalize(event["quote"]))
        if key in found:   # the same words from the same source as a signal: one claim, which remembers the class
            found[key] = replace(found[key], event_class=event["class"])
            continue
        found[key] = Claim(ref=event["class"], kind="event", direction=event["direction"], quote=event["quote"], source=event.get("source"), chunk_id=chunk_id, event_class=event["class"])
    claims, seen = [], {}
    for claim in found.values():
        seen[claim.ref] = seen.get(claim.ref, 0) + 1
        claims.append(claim if seen[claim.ref] == 1 else replace(claim, ref=f"{claim.ref}#{seen[claim.ref]}"))
    return claims


def supporting(claims: list[Claim], direction: str) -> list[Claim]:
    """The claims on the side of the stance: the ones the Critic asks the model about (counter-evidence stays as it is)."""
    return [c for c in claims if c.direction == direction]


def stance_direction(rec: Recommendation) -> str:
    """The direction a recommendation argues: bullish, always (the Advisor recommends purchases only and the Critic judges nothing else: entry 105)."""
    return "bullish"
