"""What the page shows, from what the desk wrote (UI v1 spec 4.3.1, 4.4, 4.5). Pure functions: no I/O, no model, no quote, no database.

    stage_states(path=[...], seconds={...}, running="critic", order=order)    # the six stages of the pipeline and the state of each
    work_order_view(order, understood="buy the 3 most recommended ...")        # the stocks on the work order, one row each, with the detail behind each
    account_view(holder, portfolio)                                           # the account, the holdings and the totals
    records_of(order, key)                                                    # the raw records of one row

THE ONLY PLACE THAT KNOWS THE SHAPE of a recommendation, a review, a decision and a trade, as the live run and the replay both feed it (so a capture and a live order read the same way). It adds no rule about what may
trade and scores nothing: every number and every word is one an agent wrote. `order` is a `WorkOrder` or its dict (a capture).

DECIMAL MONEY (design choices entry 112): the account's money, prices and share counts in these views are `Decimal` (a holding's value is `trade_amount`, the cash and the totals are exact sums of cents); the records of a capture hold the numbers `json`
read (an int or a float) and pass through as they are. The page receives NUMBERS: the one encoder (`money.jsonable`, `money.dumps`) writes a Decimal as an integer when it is whole and by its own digits otherwise, at the edge of the bridge (`app.py`).
"""
from datetime import date
from decimal import Decimal

from prometheus.agents.critic.report import DECISION_WORDS, LABELS, analysts_line
from prometheus.common import shares as sh
from prometheus.common.portfolio import trade_amount
from prometheus.application.recommendations import KEY_NUMBERS, _number

# ---------------------------------------------------------------- the six stages

STAGES = [
    {"key": "work_order", "label": "Create Work Order", "agent": "account_manager", "agent_label": "Account Manager", "nodes": ("account_manager",)},
    {"key": "gather", "label": "Gather Data", "agent": "data_clerk", "agent_label": "Data Clerk", "nodes": ("data_clerk",)},
    {"key": "recommend", "label": "Create Recommendations", "agent": "advisor", "agent_label": "Advisor", "nodes": ("advisor",)},
    {"key": "critique", "label": "Critique · Approve / Deny", "agent": "critic", "agent_label": "Critic", "nodes": ("critic",)},
    {"key": "prepare", "label": "Prepare Trading Order", "agent": "account_manager", "agent_label": "Account Manager", "nodes": ("account_manager_decision", "account_manager_direct", "account_manager_funds")},
    {"key": "trade", "label": "Trade · Place Order", "agent": "trader", "agent_label": "Trader", "nodes": ("trader",)},
]
STAGE_OF_NODE = {node: stage["key"] for stage in STAGES for node in stage["nodes"]}
# What a node writes on the order's `errors` when its agent could not start, decide or prepare (the desk's own words, `langgraph/desk.py`): the stage is then failed.
COULD_NOT = {"data_clerk": "the Data Clerk could not", "advisor": "the Advisor could not", "critic": "the Critic could not", "account_manager_decision": "the Account Manager could not",
             "account_manager_direct": "the Account Manager could not", "account_manager_funds": "the Account Manager could not", "trader": "the Trader could not"}
BAR_NAMES = {"news": "News", "performance": "Performance", "historical": "Historical", "filings": "Filings"}
ORDER_WORDS = {"marketable_limit": "Limit", "limit": "Limit", "market": "Market"}


def screened(text: str, guard, account_id=None) -> str:
    """A message the page is about to show, through the security layer (entry 116): a secret and another account's number are replaced by the policy's placeholder, the rest is shown as it is. With no guard (a replay of a capture) the text is as it was."""
    return text if guard is None or not isinstance(text, str) else guard.screen_message(text, account_id=account_id).text


def stage_list() -> list[dict]:
    """The stages for `run_started`: key, label, agent and the agent's name."""
    return [{k: s[k] for k in ("key", "label", "agent", "agent_label")} for s in STAGES]


def _get(order, name, default=None):
    return order.get(name, default) if isinstance(order, dict) else getattr(order, name, default)


def stage_states(*, path: list[str], seconds: dict | None = None, running: str | None = None, waiting: bool = False, failed: set[str] | None = None, order=None, request_errors: list[str] | None = None) -> list[dict]:
    """The six stages, each `{key, state, node, seconds}` with a state of pending, busy, waiting, done, skipped or failed (spec 4.3.1).

    `path` is the nodes that finished, `running` the node that is running now (a question open: `waiting`), `failed` the nodes that raised, `order` the work order so far and `request_errors` the desk's own `errors` (the request could not be read).
    A stage is FAILED when its node raised; or it is the first stage and the desk has errors; or it is `gather` and the order is `data_failed`; or its node wrote that its agent "could not" start, decide or prepare. A stage that never ran,
    with a later one that did or is running, is SKIPPED (a direct order skips three, a neutral pass skips the Critic's); the stages after the last one that ran are PENDING (the page says "not reached")."""
    seconds, failed, errors = seconds or {}, failed or set(), [e for e in (_get(order, "errors") or []) if isinstance(e, str)]
    status = _get(order, "status")
    ran = {STAGE_OF_NODE[n] for n in path if n in STAGE_OF_NODE}
    now = STAGE_OF_NODE.get(running) if running else None
    reached = max([i for i, s in enumerate(STAGES) if s["key"] in ran or s["key"] == now], default=-1)
    out = []
    for i, stage in enumerate(STAGES):
        nodes = [n for n in stage["nodes"] if n in path]
        node = nodes[-1] if nodes else (running if now == stage["key"] else stage["nodes"][0])
        took = round(sum(seconds.get(n, 0) for n in nodes), 1) if nodes else 0.0
        if stage["key"] == now:
            state = "waiting" if waiting else "busy"
        elif stage["key"] in ran:
            state = "done"
            if any(n in failed for n in stage["nodes"]) or any(e.startswith(COULD_NOT[n]) for n in nodes for e in errors if n in COULD_NOT):
                state = "failed"
            if stage["key"] == "work_order" and request_errors:
                state = "failed"
            if stage["key"] == "gather" and status == "data_failed":
                state = "failed"
        elif i < reached:
            state = "skipped"
        elif stage["key"] == "work_order" and request_errors and not path and now is None:
            state = "failed"        # the first node ran and the desk stopped there with errors (it never reports a finished node)
        elif any(n in failed for n in stage["nodes"]):
            state = "failed"
        else:
            state = "pending"
        out.append({"key": stage["key"], "state": state, "node": node, "seconds": took})
    return out


# ---------------------------------------------------------------- the account

def account_view(holder, portfolio, *, as_of: str | None = None, read_only: bool | None = None) -> dict:
    """The account (spec 4.5): who, the holdings with their value (shares x average cost: the book knows no market price), the cash and the totals. `holder` is a `UserAccount` (or anything with its fields).

    WAITING ORDERS (design choices entry 140): `cash` is ALL the cash of the account and `available` the part that may be spent (`reserved_cash` is what waiting orders have set aside); a position carries `reserved`, the shares a waiting sale has
    set aside. The total is the whole cash and the whole holdings, so a reservation is never a change of the account's value (`portfolio_changes` compares the whole cash)."""
    positions = [{"symbol": p.symbol, "name": p.name, "shares": p.shares, "cost_basis": p.cost_basis, "value": trade_amount(p.shares, p.cost_basis), "reserved": portfolio.reserved_shares.get(p.symbol, Decimal(0))} for p in portfolio.positions]
    invested = sum((p["value"] for p in positions), Decimal("0.00"))
    cash = portfolio.cash_total
    sample = getattr(holder, "seed_key", None) is not None
    return {"account": {"id": holder.account_id, "name": holder.display_name, "age": holder.age, "risk": holder.risk_tolerance, "mode": holder.trade_mode, "sample": sample, "read_only": sample if read_only is None else read_only},
            "positions": positions, "cash": cash, "available": portfolio.cash_avail, "reserved_cash": portfolio.reserved_cash, "invested": invested, "total": invested + cash, "as_of_utc": as_of}


def portfolio_changes(before: dict, after: dict) -> dict:
    """`changes: {SYMBOL: +shares}` and `cash_change` of an account view against the one at the start of the run."""
    held = {p["symbol"]: p["shares"] for p in before["positions"]}
    now = {p["symbol"]: p["shares"] for p in after["positions"]}
    return {"changes": {s: sh.difference(now.get(s, 0), held.get(s, 0)) for s in sorted(set(held) | set(now)) if not sh.same(now.get(s, 0), held.get(s, 0))},      # exact: a float `!=` would report noise as a change
            "cash_change": after["cash"] - before["cash"], "cash_before": before["cash"]}


# ---------------------------------------------------------------- the work order

def _by(items: list[dict], key: str = "record_id") -> dict:
    return {i[key]: i for i in items if isinstance(i, dict) and i.get(key)}


def _usd(value) -> str:
    return "n/a" if value is None else f"${value:,.2f}"


def _date_words(text: str | None) -> str | None:
    try:
        moment = date.fromisoformat(text[:10])
    except (TypeError, ValueError):
        return text
    return f"{moment:%b} {moment.day}, {moment.year}"


def _advisor_cell(source: dict) -> dict:
    return {"score": source.get("score"), "stance": source.get("stance"), "reason": (source.get("action") or {}).get("reason") if "action" in source else source.get("reason")}


def _critic_cell(review: dict | None) -> dict | None:
    return None if review is None else {"approved": review["approved"], "score": review["score"], "grade": review["grade"]}


def _decision_cell(decision: dict | None) -> dict | None:
    if decision is None:
        return None
    return {"will_trade": decision["will_trade"], "words": "WAITING (a day order: its cash or shares are set aside)" if decision.get("open_order_id") is not None else DECISION_WORDS.get(decision["will_trade"], decision["will_trade"]), "action": decision.get("action"), "shares": decision.get("shares"),
            "est_cost": decision.get("est_cost"), "dollars": decision.get("dollars"), "price": decision.get("price"), "origin": decision.get("origin")}


def _trade_cell(trade: dict | None) -> dict | None:
    if trade is None:
        return None
    fill = trade.get("fill") or {}
    return {"status": trade["status"], "side": trade.get("side"), "shares": fill.get("shares", trade.get("shares")), "price": fill.get("price"), "amount": fill.get("amount")}


def _order_part(decision: dict | None, trade: dict | None) -> dict | None:
    """The decision (words, shares, cost, reason, mode, balance) and the trade (status, reason, fill), for the panel's last section."""
    if decision is None and trade is None:
        return None
    out = {}
    if decision is not None:
        out["decision"] = {**_decision_cell(decision), "reason": decision.get("reason"), "mode": decision.get("mode"), "price": decision.get("price"),
                           "balance": decision.get("balance"), "binding": (decision.get("size") or {}).get("binding"), "allocation": (decision.get("size") or {}).get("allocation")}
    if trade is not None:
        fill = trade.get("fill") or {}
        out["trade"] = {**_trade_cell(trade), "trade_id": trade.get("trade_id"), "reason": trade.get("reason"), "reject_code": trade.get("reject_code"), "order_type": trade.get("order_type"), "mode": trade.get("mode"),
                        "slippage_bps": fill.get("slippage_bps"), "price_source": fill.get("price_source"), "booked": trade.get("booked")}
    return out


def _strategies(record: dict) -> list[dict]:
    return [{"name": f["family"], "passed": bool(f["passed"]), "score": f.get("score"), "note": (f.get("notes") or [None])[0] if not f["passed"] else None} for f in record.get("families") or []]


def _score_build(blend: dict | None) -> list[dict]:
    if not blend:
        return []
    weights, components = blend.get("weights") or {}, blend.get("components") or {}
    return [{"name": name, "label": BAR_NAMES[name], "weight": round(100 * weights.get(name, c["weight"])), "value": c["value"], "available": c["available"], "contribution": c["contribution"]}
            for name, c in ((n, components[n]) for n in BAR_NAMES if n in components)]


def _key_numbers(features: dict | None) -> list[dict]:
    features = features or {}
    rows = []
    for label, key, kind in KEY_NUMBERS:
        name = "vol_63d_ann" if key == "vol_ann" and key not in features else key
        value = (features.get(name) or {}).get("value")
        if value is not None and not isinstance(value, bool):
            rows.append({"label": label, "value": _number(value, kind)})
    return rows


def _plan(trade: dict | None) -> dict | None:
    if not trade:
        return None
    zone = trade.get("entry_zone")
    style = trade.get("order_style")
    return {"entry_zone": f"{_usd(zone[0])} – {_usd(zone[1])}" if zone else None, "stop": _usd(trade["stop_price"]) if trade.get("stop_price") is not None else None,
            "target": _usd(trade["target_price"]) if trade.get("target_price") is not None else None,
            "order": (ORDER_WORDS.get(style, (style or "").replace("_", " ").capitalize()) + (f" {_usd(trade['limit_price'])}" if trade.get("limit_price") is not None else "")) if style else None,
            "time_stop": _date_words(trade.get("time_stop_date")), "horizon": f"{trade['horizon_bars']} sessions" if trade.get("horizon_bars") else None,
            "invalidation": [f"{i.get('kind', '').replace('_', ' ')}: {i['value'] if not isinstance(i.get('value'), list) else ', '.join(map(str, i['value']))}" for i in trade.get("invalidation") or []],
            "notes": list(trade.get("notes") or [])}


def _review_part(review: dict | None) -> dict | None:
    if review is None:
        return None
    risk = review.get("risk") or {}
    return {"approved": review["approved"], "score": review["score"], "grade": review["grade"], "summary": review.get("summary"),
            "dimensions": [{"name": n, "label": LABELS.get(n, n).capitalize(), "points": d["points"], "weight": d["weight"], "applicable": d.get("applicable", True)} for n, d in (review.get("dimensions") or {}).items()],
            "blockers": [{"code": b["code"], "message": b["message"]} for b in review.get("blockers") or []],
            "warnings": [{"code": w["code"], "message": w["message"]} for w in review.get("warnings") or []],
            "risk": {"level": risk.get("risk_level"), "score": risk.get("risk_score"), "tolerance": risk.get("tolerance_stated"), "allowed": risk.get("allowed_level")} if risk else None,
            "groups": [{"name": g, "label": g.capitalize(), "bad": bool(codes), "codes": list(codes)} for g, codes in (review.get("groups") or {}).items()],      # Critic v3: the three groups of judgment, bad (a sign of it fired) or clean; [] on a review before version 3
            "analysts": analysts_line(review) if review.get("analysts") else None,                                                                    # Critic v3: the analysts' counts in one line (the Critic's own sentence); None on a review before version 3
            "pattern_bonus": review.get("pattern_bonus") or 0,
            "valid_until_utc": review.get("valid_until_utc"), "requires_human": bool(review.get("requires_human"))}


def _recommended_detail(record: dict, review: dict | None, decision: dict | None, trade: dict | None, of: int | None) -> dict:
    thesis = record.get("thesis") or {}
    ranking = record.get("ranking") or {}
    return {"header": {"score": record.get("score"), "stance": record.get("stance"), "rank": ranking.get("rank"), "of": of, "strategy": (record.get("trade") or {}).get("family")},
            "thesis": thesis.get("thesis"), "bull": [{"text": c["text"], "ref": c.get("ref")} for c in thesis.get("bull_case") or []], "bear": [{"text": c["text"], "ref": c.get("ref")} for c in thesis.get("bear_case") or []],
            "evidence": [{"source": s.get("source"), "direction": s.get("direction"), "quote": s.get("quote"), "verified": s.get("verified"), "counted": s.get("counted"), "reason": s.get("reason")} for s in record.get("signals") or []],
            "score_build": _score_build(record.get("blend")), "key_numbers": _key_numbers(record.get("features")), "plan": _plan(record.get("trade")), "review": _review_part(review), "order": _order_part(decision, trade)}


def _row(key: str, symbol: str, company: str | None, rank, group: str, **parts) -> dict:
    return {"key": key, "symbol": symbol, "company": company, "rank": rank, "group": group, "advisor": None, "critic": None, "decision": None, "trade": None, "detail": {}, **parts}


def work_order_view(order, *, understood: str | None = None, found: list[dict] | None = None) -> dict:
    """The stocks on the work order (spec 4.4): direct orders first, then the recommended by rank, the neutral passes, the scored and not recommended, and, before the Advisor has answered, the companies the Data Clerk found.
    `found` is [{symbol, company}] for that last group (the live run knows the company names; a capture has the subject only). A trade that matches no row gets a row of its own."""
    recs = sorted(_get(order, "recommendations") or [], key=lambda r: ((r.get("ranking") or {}).get("rank") is None, (r.get("ranking") or {}).get("rank") or 0))
    candidates, reviews = _get(order, "candidates") or [], _by(_get(order, "reviews") or [])
    decisions, directs = _by(_get(order, "trade_decisions") or []), _get(order, "direct_orders") or []
    trades = {}
    for t in _get(order, "trades") or []:
        trades.setdefault(t.get("record_id"), t)
        if t.get("status") == "filled":
            trades[t.get("record_id")] = t
    rows, used = [], set()

    for d in directs:                                   # 1. the user's own orders
        rid = d["record_id"]
        used.add(rid)
        decision, trade = decisions.get(rid), trades.get(rid)
        rows.append(_row(rid, d["symbol"], d.get("company"), None, "direct", decision=_decision_cell(decision), trade=_trade_cell(trade),
                         detail={"header": {"score": None, "stance": None, "rank": None, "of": None, "strategy": None},
                                 "message": "Your own order: no recommendation is made and the Critic is not asked.",
                                 "your_order": (f"{d['side']} ${decision['dollars']:,.2f} of {d['symbol']} (about {sh.text(d['shares'])} {sh.noun(d['shares'])})" if decision and decision.get("dollars") else      # an order in dollars says its dollars first (entry 111)
                                               f"{d['side']} {sh.text(d['shares'])} {sh.noun(d['shares'])} of {d['symbol']}" if d.get("shares") else f"{d['side']} {d['symbol']}"), "order": _order_part(decision, trade)}))
    scored_count = len(recs) + len(candidates)
    for r in recs:                                      # 2. the recommendations, by rank
        rid = r["identity"]["record_id"]
        used.add(rid)
        review, decision, trade = reviews.get(rid), decisions.get(rid), trades.get(rid)
        rows.append(_row(r["symbol"], r["symbol"], r.get("company"), (r.get("ranking") or {}).get("rank"), "recommended", advisor=_advisor_cell(r), critic=_critic_cell(review),
                         decision=_decision_cell(decision), trade=_trade_cell(trade), detail=_recommended_detail(r, review, decision, trade, scored_count)))
    neutral = {d["symbol"]: d for d in decisions.values() if d.get("origin") == "neutral"}
    shown = {r["symbol"] for r in recs}
    for c in sorted(candidates, key=lambda c: (c.get("rank") is None, c.get("rank") or 0)):
        symbol = c["symbol"]
        if symbol in shown:
            continue
        shown.add(symbol)
        advisor = {"score": c.get("score"), "stance": c.get("stance"), "reason": c.get("reason")}
        header = {"score": c.get("score"), "stance": c.get("stance"), "rank": c.get("rank"), "of": scored_count, "strategy": None}
        if symbol in neutral:                           # 3. a neutral pass: the Advisor found no strategy and it counts as approved
            decision = neutral[symbol]
            used.add(decision["record_id"])
            trade = trades.get(decision["record_id"])
            rows.append(_row(symbol, symbol, c.get("company"), c.get("rank"), "neutral", advisor=advisor, decision=_decision_cell(decision), trade=_trade_cell(trade),
                             detail={"header": header, "message": "The Advisor found no strategy for this company (a neutral score). A neutral recommendation counts as approved, so there is no Critic review.",
                                     "strategies": _strategies(c), "score_build": _score_build(c.get("blend")), "order": _order_part(decision, trade)}))
        else:                                           # 4. scored and not recommended
            rows.append(_row(symbol, symbol, c.get("company"), c.get("rank"), "scored", advisor=advisor, reason="not recommended: " + (c.get("reason") or "no reason was recorded"),
                             detail={"header": header, "message": "Not recommended: " + (c.get("reason") or "no reason was recorded"), "strategies": _strategies(c), "score_build": _score_build(c.get("blend"))}))
    if not recs and not candidates and not directs:     # 5. the Advisor has not answered yet: the companies the Data Clerk found
        subject = _get(order, "subject")
        names = found if found is not None else [{"symbol": s, "company": None} for s in (subject if isinstance(subject, list) else [])]
        rows += [_row(n["symbol"], n["symbol"], n.get("company"), None, "found", detail={"message": "Found by the Data Clerk. The Advisor has not answered yet."}) for n in names if n.get("symbol")]
    for t in _get(order, "trades") or []:               # a trade that matches no row
        if t.get("record_id") not in used:
            used.add(t.get("record_id"))
            rows.append(_row(t["symbol"], t["symbol"], None, None, "trade", trade=_trade_cell(t), detail={"order": _order_part(None, t)}))
    seen, unique = {}, []
    for row in rows:                                    # one key per row (the page selects by it)
        seen[row["key"]] = seen.get(row["key"], 0) + 1
        if seen[row["key"]] > 1:
            row["key"] = f"{row['key']}#{seen[row['key']]}"
        unique.append(row)
    return {"request_id": _get(order, "request_id"), "status": _get(order, "status"), "kind": "direct" if directs or _get(order, "action_type") == "trade" else "research", "understood": understood, "rows": unique}


def records_of(order, key: str) -> dict | None:
    """The raw records behind one row (`GET /records/{key}`): whichever of the recommendation, the review, the decision, the trade, the candidate and the direct order exist. None when no row has that key."""
    base = key.split("#")[0]
    view = work_order_view(order)
    if not any(r["key"] == key for r in view["rows"]):
        return None
    direct = next((d for d in _get(order, "direct_orders") or [] if d["record_id"] == base), None)
    rec = next((r for r in _get(order, "recommendations") or [] if r["symbol"] == base), None)
    rid = (rec or {}).get("identity", {}).get("record_id") or (direct or {}).get("record_id")
    cand = next((c for c in _get(order, "candidates") or [] if c["symbol"] == base), None)
    decision = next((d for d in _get(order, "trade_decisions") or [] if (rid and d["record_id"] == rid) or (not rid and d["symbol"] == base) or (cand and d.get("origin") == "neutral" and d["symbol"] == base)), None)
    rid = rid or (decision or {}).get("record_id")
    review = next((r for r in _get(order, "reviews") or [] if rid and r["record_id"] == rid), None)
    trade = next((t for t in _get(order, "trades") or [] if (rid and t.get("record_id") == rid) or (not rid and t["symbol"] == base)), None)
    found = {"recommendation": rec, "review": review, "decision": decision, "trade": trade, "candidate": cand, "direct_order": direct}
    return {k: v for k, v in found.items() if v is not None}


def order_snapshot(order) -> dict:
    """A plain dict of the order without its (large) data: what `records_of` and the page read."""
    return order if isinstance(order, dict) else order.model_dump(exclude={"data"})


def companies_found(order, plan: dict | None = None) -> list[dict]:
    """[{symbol, company}] the Data Clerk found: from the desk's plan (a research run), else from the order's data (a direct order keeps it)."""
    result = (plan or {}).get("result")
    listed = getattr(result, "companies", None) or []
    if not listed and _get(order, "data") is not None:
        listed = _get(_get(order, "data"), "companies") or []
    return [{"symbol": c.get("symbol"), "company": c.get("company")} for c in listed if c.get("symbol")]


WRITTEN_BY = {"advisor_runs": ("advisor",), "recommendations": ("advisor",), "candidates": ("advisor",), "critic_runs": ("critic",), "reviews": ("critic",),
              "trade_decisions": ("account_manager_decision", "account_manager_direct", "account_manager_funds"), "trades": ("trader",), "trader_runs": ("trader",)}
STATUS_AFTER = {"account_manager": "new", "data_clerk": "data_ready", "advisor": "recommended", "critic": "reviewed", "account_manager_decision": "decided", "account_manager_direct": "decided", "account_manager_funds": "decided"}


def cut_order(order: dict, path: list[str]) -> dict:
    """A copy of a captured order cut back to what the nodes of `path` had written (the replay, spec 4.6): after the Data Clerk the subject only, after the Advisor the recommendations and the candidates, after the Critic
    the reviews, after the decision the decisions, after the Trader the trades. A direct order keeps its own records (`direct_orders`, `data`) from the first node on."""
    cut = dict(order)
    for field, writers in WRITTEN_BY.items():
        if not any(w in path for w in writers):
            cut[field] = []
    if "data_clerk" not in path and not order.get("direct_orders"):
        cut["subject"] = []
    cut["status"] = order.get("status") if "trader" in path else STATUS_AFTER.get(path[-1], "new") if path else "new"
    return cut


# ---------------------------------------------------------------- how a run ended

def waiting_lines(order) -> list[str]:
    """The sentence of each direct order that was placed to WAIT for its limit price (a decision with an `open_order_id`, design choices entry 140): "Waiting: buy 10 MU at $90.00 or less (now $95.12), $900.00 set aside, until 4:00 pm ET today.". Empty when none waits."""
    decisions = (_get(order, "trade_decisions") or []) if order is not None else []
    return [d["reason"] for d in decisions if isinstance(d, dict) and d.get("open_order_id") is not None and isinstance(d.get("reason"), str)]


def run_verdict(before: dict, after: dict, *, action: str | None, order=None, result=None, mode: str | None = None) -> dict:
    """The verdict of a run, as the session decides it (`session.run_session`): `traded` when the portfolio changed (cash or shares), `looked` when the request was only to look, otherwise `no_trade` with the reasons the
    session gives (`session.why_not`). `before` and `after` are account views. `stopped` and `failed` are the caller's."""
    from prometheus.application.session import why_not

    moved = portfolio_changes(before, after)
    waiting = waiting_lines(order)      # direct orders that now WAIT for their limit price (entry 140): the run's last message says so, with what is set aside and until when
    if moved["changes"] or moved["cash_change"] != 0:
        names = ", ".join(f"{sh.signed(shares)} {symbol}".replace("-", "−") for symbol, shares in moved["changes"].items())
        return {"status": "traded", "headline": f"The portfolio changed: {names}. Cash ${before['cash']:,.2f} → ${after['cash']:,.2f}." if names else f"Cash ${before['cash']:,.2f} → ${after['cash']:,.2f}.", "reasons": waiting}
    if waiting:
        return {"status": "waiting", "headline": waiting[0], "reasons": waiting[1:]}
    if action not in ("buy", "order"):
        return {"status": "looked", "headline": "A look only: nothing was to be bought. The portfolio is as it was.", "reasons": []}
    reasons = why_not(result, order) if order is not None else ["no work order was made"]
    return {"status": "no_trade", "headline": "Nothing was traded: " + reasons[0], "reasons": reasons}
