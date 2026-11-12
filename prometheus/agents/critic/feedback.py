"""The Critic's feedback, as metadata of the recommendation it judged (Critic v1 Amendment 2, design choices entry 64). Plain data from a review's own numbers: no model writes a field.

    attach(order, config)            after the review: each recommendation of the order gets a `critic` block, joined to its review by `record_id`
    attach_decisions(order)          after the Account Manager decides: each block gets the trade `decision` (yes, no, pending) with its reason

The block is for TWO readers who do not read the whole review: the person who is told why a recommendation was approved or denied, and a screen that draws it (bars for the dimensions, a gauge
for the score, a list of chart patterns, a badge for the verdict). So it is flat, ordered, JSON, and carries what a picture needs: the dimensions in a fixed order with their weight, score,
points, share of the weight earned and a `band` (strong, ok, weak) for a colour; the FINAL GATE spelled out (the blended score, the mark, the cap, what blocked it); the patterns; the findings; the
sizes; the decision. The full review stays on `order.reviews` and is linked by `review_id`.

THE FINAL GATE. A recommendation is approved when it has no blocker and its BLENDED score reaches the mark:

    base   = sum(weight x dimension score) / sum(weights of the dimensions that apply)          six dimensions: accuracy, evidence, risk fit, timing, value, chart
    score  = min(100, round(base + the great-deal bonus));  with a blocker, at most `blocked_score_cap` (49)
    approved = no blocker and score >= `approve_min_score` (70)
"""
from prometheus.agents.critic import report
from prometheus.agents.critic.config import DIMENSIONS, CriticConfig

SCHEMA = 1
TITLES = {"accuracy": "Accuracy", "evidence": "Evidence", "risk_fit": "Risk fit", "timing": "Timing", "value": "Value", "chart": "Chart"}
HEADLINES = {"accuracy": lambda review, record: report._accuracy(review), "evidence": lambda review, record: report._evidence(review), "risk_fit": lambda review, record: report._risk(review),
             "timing": report._timing, "value": lambda review, record: report._value(review), "chart": lambda review, record: report._chart(review)}
FORMULA = "score = min(100, round(sum(weight x dimension score) / sum(weights that apply) + bonus)); a blocker caps it; approved = no blocker and score >= the mark"


def band(score: float, config: CriticConfig) -> str:
    """`strong` from the mark of a B (80), `ok` from the mark of a D (60), else `weak`: the colour of a bar."""
    grades = config.verdict.grades
    return "strong" if score >= grades["B"] else "ok" if score >= grades["D"] else "weak"


def _finding(f: dict) -> dict:
    return {"code": f["code"], "dimension": f["dimension"], "message": f["message"], "requires_human": f["requires_human"]}


def feedback(review: dict, record: dict | None, config: CriticConfig) -> dict:
    """The `critic` block of one recommendation, from its review (a dict of `order.reviews`) and the record. `decision` is filled later, by `attach_decisions`."""
    v = config.verdict
    dimensions = review["dimensions"]
    applicable = [n for n in DIMENSIONS if dimensions[n]["applicable"]]
    weight_total = sum(dimensions[n]["weight"] for n in applicable)
    base = round(sum(dimensions[n]["weight"] * dimensions[n]["score"] for n in applicable) / weight_total, 2) if weight_total else 0.0
    rows = []
    for name in DIMENSIONS:
        d = dimensions[name]
        rows.append({"name": name, "title": TITLES[name], "applicable": d["applicable"], "weight": d["weight"], "score": d["score"] if d["applicable"] else None,
                     "points": d["points"] if d["applicable"] else None, "share": round(d["score"] / 100, 4) if d["applicable"] else None,
                     "weight_share": round(d["weight"] / weight_total, 4) if d["applicable"] and weight_total else None,
                     "band": band(d["score"], config) if d["applicable"] else None, "headline": HEADLINES[name](review, record or {})})
    blockers = [_finding(f) for f in review["blockers"]]
    return {
        "schema": SCHEMA, "review_id": review["review_id"], "reviewed_utc": review["reviewed_utc"], "valid_until_utc": review["valid_until_utc"],
        "verdict": {"approved": review["approved"], "decision": "approve" if review["approved"] else "deny", "score": review["score"], "grade": review["grade"], "band": band(review["score"], config),
                    "summary": review["summary"]},
        "gate": {"formula": FORMULA, "base_score": base, "bonus": review["bonus"], "weight_total": weight_total, "score_before_blockers": review["score_before_blockers"], "score": review["score"],
                 "approve_min_score": v.approve_min_score, "auto_min_score": v.auto_min_score, "blocked_score_cap": v.blocked_score_cap,
                 "passed_score": review["score_before_blockers"] >= v.approve_min_score, "blocked": bool(blockers), "blocked_by": [b["code"] for b in blockers],
                 "denial_reasons": review["denial_reasons"], "margin": review["score"] - v.approve_min_score},
        "dimensions": rows,
        "patterns": {"holding": review["chart"]["patterns"], "unknown": review["chart"]["unknown"], "points": review["chart"]["points"]},
        "findings": {"blockers": blockers, "warnings": [_finding(f) for f in review["warnings"]], "notes": [_finding(f) for f in review["notes"]]},
        "size": {"suggested_shares": review["suggested_shares"], "approved_shares": review["approved_shares"], "allowed_shares": review["risk"].get("allowed_shares")},
        "risk": {"level": review["risk"].get("risk_level"), "score": review["risk"].get("risk_score"), "allowed_level": review["risk"].get("allowed_level")},
        "value": {"deal": review["value"].get("deal"), "premium": review["value"].get("premium"), "discount": review["value"].get("discount"), "reward_to_risk": review["value"].get("reward_to_risk")},
        "needs_human": {"required": review["requires_human"], "reasons": review["requires_human_reasons"]},
        "audit": {"critic_model": review["audit"]["critic_model"], "independent_model": review["audit"]["independent_model"], "config_version": review["audit"]["config_version"]},
        "decision": None,
    }


def attach(order, config: CriticConfig) -> None:
    """Put the `critic` block on each recommendation that has a review (joined by `record_id`). Nothing else of a record changes: the Advisor's fields are not touched."""
    records = {(r.get("identity") or {}).get("record_id"): r for r in order.recommendations}
    for review in order.reviews:
        record = records.get(review["record_id"])
        if record is not None:
            record["critic"] = feedback(review, record, config)


def attach_decisions(order) -> None:
    """Put the Account Manager's decision into the `critic` block of its recommendation: what will be traded and why, and whether the user was asked."""
    records = {(r.get("identity") or {}).get("record_id"): r for r in order.recommendations}
    for d in order.trade_decisions:
        block = (records.get(d["record_id"]) or {}).get("critic")
        if block is not None:
            block["decision"] = {"will_trade": d["will_trade"], "mode": d["mode"], "shares": d["shares"], "est_cost": d["est_cost"], "asked_user": d["asked_user"], "escalated": d["escalated"],
                                 "reason": d["reason"], "decided_utc": d["decided_utc"]}
