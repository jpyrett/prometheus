"""What the user reads: the Critic's scorecard (Critic v1 spec section 13.3). Plain text from a review's own numbers: no model writes a line.

    card = scorecard(review, record)        # {"text": the lines, ...}: the Ask prompt shows it, and so does the report
    lines = review_section(order)           # the whole CRITIC REVIEW section of a report, and the TRADE DECISIONS line (with the size the Account Manager decided)

`review` is a review as it lives on `order.reviews` (a dict); `record` is the recommendation it reviews (a dict of `order.recommendations`: the card quotes its rank, company and entry zone). The card has no
size: the Critic does not decide how much to buy (the Account Manager does, and adds what it would place to the card it shows the user: `decisions._card`).
"""
from prometheus.agents.critic.common import EASTERN, pct
from prometheus.agents.critic.reasons import GROUP_WORDS
from prometheus.common import shares as sh
from prometheus.common.work_order import parse_utc

DECISION_WORDS = {"yes_auto": "yes (AUTO)", "yes_user": "yes (you approved)", "pending_user": "pending your answer (ASK)", "no_system_denied": "no (system denied)",
                  "no_expired": "no (the approval ran out)", "no_funds": "no (no funds)", "no_user_denied": "no (you denied)"}
LABELS = {"accuracy": "accuracy", "evidence": "evidence", "risk_fit": "risk fit", "timing": "timing", "value": "value", "chart": "chart", "analysts": "analysts"}
GROUP_LABELS = {"news": "news", "performance": "performance", "price": "price"}


def _points(review: dict, name: str) -> str:
    d = review["dimensions"][name]
    return f"{d['points']:.0f}/{d['weight']:g}"


def _et(text: str | None) -> str:
    moment = parse_utc(text)
    return "?" if moment is None else moment.astimezone(EASTERN).strftime("%H:%M ET")


def _accuracy(review: dict) -> str:
    a = review["accuracy"]
    matched = sum(1 for r in a["rechecked"] if r["match"])
    parts = [f"{matched} of {len(a['rechecked'])} values match what was fetched again" if a["rechecked"] else "no values could be compared"]
    sources = a["sources"]
    parts.append(f"{sum(1 for s in sources if s['found'] and s['quote_in_text'])} of {len(sources)} sources read again" if sources else "no quotes to read again")
    for check in a["cross"]:
        parts.append({"price": "the close agrees with a second provider" if check["passed"] else "the close DISAGREES with a second provider",
                      "identity": "the symbol is in the directory" if check["passed"] else "the symbol or its name does not check out",
                      "high_52w": "the 52-week high agrees with Finnhub's" if check["passed"] else "the 52-week high differs from Finnhub's"}[check["name"]])
    if a["integrity"]["failed"]:
        parts.append("the record does not add up: " + ", ".join(a["integrity"]["failed"][:3]))
    return "; ".join(parts)


def _evidence(review: dict) -> str:
    e = review["evidence"]
    counts: dict[str, int] = {}
    for claim in e["claims"]:
        counts[claim["label"]] = counts.get(claim["label"], 0) + 1
    if not e["claims"]:
        parts = ["no quotes to review"]
    else:
        parts = [f"{counts.get('supports', 0)} of {len(e['claims'])} quotes support the claim" + "".join(f", {n} {label}" for label, n in counts.items() if label != "supports")]
    if e["control"] and not e["control"]["passed"]:
        parts.append("the model failed its reading check: its labels were not used")
    bad = [f for f in e["fresh"] if f["label"] == "bearish" or f["classes"]]
    parts.append(f"{len(bad)} newer item(s) look bad" if bad else f"no newer bad news in {len(e['fresh'])} fresh sources" if e["fresh"] else "no newer sources")
    return "; ".join(parts)


def _risk(review: dict) -> str:
    r = review["risk"]
    if r["risk_score"] is None:
        return "could not be judged"
    text = f"risk {r['risk_level']} ({r['risk_score']:g}), your limit is {r['allowed_level']}"
    if r["tolerance_effective"] != r["tolerance_stated"]:
        text += f" (tolerance lowered to {r['tolerance_effective']})"
    return text


def _timing(review: dict, record: dict | None) -> str:
    t = review["timing"]
    text = f"{t['session'].replace('_', '-')} session" if t["session"] != "closed" else f"market closed ({t['session_reason']})"
    zone = ((record or {}).get("trade") or {}).get("entry_zone")
    if t["deal_price"] is not None and zone:
        place = "inside" if 0 <= (t["zone_position"] or 0) <= 1 else "above" if (t["zone_position"] or 0) > 1 else "under"
        text += f"; {t['deal_price']:.2f} is {place} the entry zone {zone[0]:.2f} to {zone[1]:.2f}"
    elif t["deal_price"] is None:
        text += "; no price"
    return text


def _value(review: dict) -> str:
    v = review["value"]
    if v["deal"] is None:
        return "could not be judged"
    gap = (f"{pct(abs(v['premium']))} {'above' if v['premium'] > 0 else 'under'} its 6-month average of {v['avg_6m']:.2f}" if v["avg_6m"] is not None and v["premium"] is not None
           else "no six-month average to judge the price against")
    return f"{v['deal'].upper()}: {gap}"


def analysts_line(review: dict) -> str:
    """The analysts' counts in one line: `10 strong buy, 12 buy, 8 hold, 1 sell, 0 strong sell in 2026-11-01 (31 analysts): mean 3.97 of 5, up 0.21 since 2026-08-01`."""
    a = review.get("analysts") or {}
    if not a.get("applicable"):
        return a.get("note") or "no analyst ratings"
    text = f"{a['note']}: mean {a['mean']:.2f} of 5"
    if a.get("mean_change") is not None:
        text += f", {'up' if a['mean_change'] > 0 else 'down' if a['mean_change'] < 0 else 'unchanged'}{'' if a['mean_change'] == 0 else ' ' + format(abs(a['mean_change']), '.2f')} since {a['compared_period']}"
    return text


def groups_line(review: dict) -> str | None:
    """The three groups of judgment in one line (`news clean, performance BAD (trend_not_confirmed), price clean`), or None for a review of a version before the groups."""
    groups = review.get("groups")
    if not groups:
        return None
    return ", ".join(f"{GROUP_LABELS[g]} " + (f"BAD ({', '.join(groups[g])})" if groups.get(g) else "clean") for g in GROUP_LABELS if g in groups)


def _chart(review: dict) -> str:
    c = review["chart"]
    if not c["facts"]:
        return "could not be judged"
    words = lambda side: ", ".join(p["name"].replace("_", " ") for p in c["patterns"] if p["direction"] == side)
    parts = [f"bullish: {words('bullish')}" if words("bullish") else "", f"bearish: {words('bearish')}" if words("bearish") else ""]
    return "; ".join(p for p in parts if p) or "no clear pattern"


def scorecard(review: dict, record: dict | None = None) -> dict:
    """The card of one review: {"text", "symbol", "approved", "score", "grade", "summary"}. `text` is the lines of the spec's example. It names no number of shares and no cost: that is the Account Manager's."""
    record = record or {}
    approved = review["approved"]
    rank = (record.get("ranking") or {}).get("rank")
    head = f"  #{rank if rank is not None else '-'}  {review['symbol']:<6} {(record.get('company') or '')[:22]:<22} {'APPROVE' if approved else 'DENY   '} score {review['score']}/100 ({review['grade']})"
    if approved:
        head += f"      open a position      valid until {_et(review['valid_until_utc'])}"
    elif review["score_before_blockers"] > review["score"]:
        head += f"      would have been {review['score_before_blockers']} without the blockers"
    lines = [head]
    rows = (("accuracy", _accuracy(review)), ("evidence", _evidence(review)), ("risk_fit", _risk(review)), ("timing", _timing(review, record)), ("value", _value(review)), ("chart", _chart(review)), ("analysts", analysts_line(review)))
    for name, text in rows:
        if name not in review["dimensions"]:      # a review of a version before the analysts has six dimensions
            continue
        if review["dimensions"][name]["applicable"]:
            lines.append(f"      {LABELS[name]:<9} {_points(review, name):>5}   {text}")
        else:
            lines.append(f"      {LABELS[name]:<9} {'-':>5}   {text}")
    if review["dimensions"]["value"]["applicable"]:
        lines.append(f"      {'bonus':<9} {'+' + format(review['bonus'], 'g'):>5}   " + (f"{pct(review['value']['discount'])} under its 6-month average" if review["bonus"] else "no discount to the 6-month average"))
    if review.get("pattern_bonus"):
        lines.append(f"      {'patterns':<9} {format(review['pattern_bonus'], '+g'):>5}   the net chart points move the score ({_chart(review)})")
    if groups_line(review):
        lines.append(f"      groups:   {groups_line(review)}")
    if review["warnings"]:
        lines.append("      warnings: " + "; ".join(dict.fromkeys(w["code"] for w in review["warnings"])))
    for b in review["blockers"]:
        lines.append(f"      BLOCKER {b['code']}: {b['message']}")
    if review["denial_reasons"] and "score_below_minimum" in review["denial_reasons"]:
        lines.append("      BLOCKER score_below_minimum: the score is under the mark for approval")
    two = [r[len("group:"):] for r in review["denial_reasons"] if r.startswith("group:")]
    if two:
        lines.append("      DENIED: bad signs in " + " and ".join(f"{GROUP_WORDS[g]} ({', '.join(review['groups'].get(g) or [])})" for g in two))
    if "needs_human_review" in review["denial_reasons"]:
        lines.append("      DENIED needs_human_review: a person must look before this can trade: " + ", ".join(review["requires_human_reasons"]))
    return {"text": "\n".join(lines), "symbol": review["symbol"], "approved": approved, "score": review["score"], "grade": review["grade"], "summary": review["summary"]}


def header(order) -> str:
    """The first line of the section: the model, whether it is independent of the Advisor's, the profile, the mode, and the moment and session of the review."""
    run = order.critic_runs[-1] if order.critic_runs else {}
    independent = {True: f"independent of the Advisor's {run.get('advisor_model')}", False: f"SAME PRODUCER as the Advisor's {run.get('advisor_model')}", None: "independence from the Advisor's model unknown"}[run.get("independent_model")]
    mode = next((d.get("mode") for d in order.trade_decisions), None)
    return (f"CRITIC REVIEW  (model {run.get('model') or 'none'}, {independent}; profile {run.get('tolerance') or '?'}; mode {mode.upper() if mode else 'n/a'}; "
            f"reviewed {run.get('now_et') or '?'}, {str(run.get('session') or '?').replace('_', '-')} session" + ("; IGNORE_MARKET_HOURS: a closed market does not deny" if run.get("ignore_market_hours") else "") + ")")


def _decided_size(decision: dict) -> str:
    """What the Account Manager decided to buy, for the line of decisions (the Critic's card has no size): ", 44 shares, about $7,352"; nothing when no share was decided."""
    if not decision.get("shares"):
        return ""
    if decision.get("dollars"):      # an order in dollars (entry 111, M2): the dollars first, the shares an estimate
        return f", ${decision['dollars']:,.0f}, about {sh.text(decision['shares'])} shares"
    return f", {sh.text(decision['shares'])} shares, about ${decision['est_cost']:,.0f}"


def decisions_line(order) -> str:
    """The Account Manager's decisions on one line: each symbol, the decision, and the size it decided."""
    return "TRADE DECISIONS  " + "    ".join(f"{d['symbol']}: {DECISION_WORDS.get(d['will_trade'], d['will_trade'])}{_decided_size(d)}" for d in order.trade_decisions) if order.trade_decisions else ""


def review_section(order) -> list[str]:
    """The CRITIC REVIEW section of a report: the header, one card per review (in the order of the recommendations), the errors of the step, and the trade decisions. Empty when nothing was reviewed."""
    if not order.reviews:
        return []
    records = {r["identity"]["record_id"]: r for r in order.recommendations if r.get("identity")}
    lines = [header(order)]
    lines += [scorecard(review, records.get(review["record_id"]))["text"] for review in order.reviews]
    run = order.critic_runs[-1] if order.critic_runs else {}
    lines += [f"  critic note: {e}" for e in [*run.get("errors", []), *run.get("dropped", [])][:6]]
    if order.trade_decisions:
        lines.append(decisions_line(order))
    return lines
