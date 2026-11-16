"""Build src/fixtures/questions.json: the three kinds of question the page can be asked, from REAL records (never typed by hand).

    ~/miniconda3/envs/cmu/bin/python ui/scripts/make_question_fixtures.py          (from the repo root; reads, imports and prints nothing else)

The replay server only ever asks the confirmation (and no capture of ours is in Ask mode with approvals), so the other two are built here the way the
live desk builds them (`prometheus/application/web/desk_run.py`, `replay.py::_ask_approvals`):
  confirm   the question event the replay recorded in ui/src/fixtures/gates_ask.json
  approve   the first of the two trades Bob Jones' capture decided (bob_jones_oil.json): the Critic's own scorecard text, the proposal, the cash
  clarify   the first question of the Account Manager's real model reply (clarify_ask_mode.json) and the facts lines `clarify.facts_lines` writes for the capture's approved purchases
  missing   the question about Chick-fil-a, a stock the Data Clerk did not find (design choices entry 102): the Account Manager's real model wording (direct_order_names.json), checked by `missing.check_wording`, and the
            stocks that stayed on the order in the real session d_pyrett_names_ask.json, put in the page's payload by `desk_run.missing_payload` (the same function the live desk uses)
Each is a `question` event; the page's dev mode (`?fixture=bob&at=29&question=approve`) and the tests read them.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO))

from prometheus.agents.account_manager.clarify import facts_lines     # noqa: E402
from prometheus.agents.critic.report import scorecard                 # noqa: E402
from prometheus.agents.account_manager import missing                 # noqa: E402
from prometheus.application.web.desk_run import missing_payload       # noqa: E402

CAPTURES = REPO / "tests" / "fixtures" / "session_real"
OUT = REPO / "ui" / "src" / "fixtures" / "questions.json"

gates = json.loads((OUT.parent / "gates_ask.json").read_text(encoding="utf-8"))
confirm = next(e for e in gates["events"] if e["type"] == "question")

bob = json.loads((CAPTURES / "bob_jones_oil.json").read_text(encoding="utf-8"))
order = bob["order"]
decisions = [d for d in order["trade_decisions"] if d["will_trade"].startswith("yes_")]
records = {r["identity"]["record_id"]: r for r in order["recommendations"]}
reviews = {r["record_id"]: r for r in order["reviews"]}
cash = order["trader_runs"][0]["portfolio_before"]["cash"]
proposal = [{"symbol": d["symbol"], "action": d["action"], "shares": d["shares"], "est_cost": d["est_cost"]} for d in decisions]
first = decisions[0]
text = scorecard(reviews[first["record_id"]], records[first["record_id"]])["text"]
approve = {"seq": 0, "type": "question", "data": {
    "question_id": "approve0001", "kind": "approve", "title": f"Approve trade 1 of {len(decisions)}", "text": text, "symbol": first["symbol"], "shares": first["shares"], "est_cost": first["est_cost"],
    "action": first["action"], "n": 1, "of": len(decisions), "cash": cash, "proposal": proposal, "scorecard": text, "ts": 0}}

robin = json.loads((CAPTURES / "bob_jones_oil_round_robin.json").read_text(encoding="utf-8"))["order"]
bought = [d for d in robin["trade_decisions"] if d["will_trade"].startswith("yes_")]
cash_rr = robin["trader_runs"][0]["portfolio_before"]["cash"]
def purchase(d: dict) -> dict:
    price = round(d["est_cost"] / d["shares"], 2)           # the decision keeps the cost of its shares: the price per share is that over the shares
    most = int(cash_rr // price)
    return {"symbol": d["symbol"], "price": price, "most_shares_your_cash_buys": most, "most_dollars": round(most * price, 2), "limited_by": None}


facts = {"cash": cash_rr, "equity": cash_rr, "approved_purchases": [purchase(d) for d in bought]}
card = json.loads((CAPTURES / "clarify_ask_mode.json").read_text(encoding="utf-8"))["scenarios"]["percent_total"]["cards"][0]
clarify = {"seq": 0, "type": "question", "data": {
    "question_id": "clarify0001", "kind": "clarify", "title": "How much?", "text": card["question"], "question": card["question"], "n": card["n"], "of": card["of"], "facts": facts_lines(facts), "ts": 0}}

names = json.loads((CAPTURES / "direct_order_names.json").read_text(encoding="utf-8"))
kept = [{"typed": r["brand"], "symbol": r["symbol"], "company": r["company"], "how": r["how"]} for r in json.loads((CAPTURES / "d_pyrett_names_ask.json").read_text(encoding="utf-8"))["order"]["data"]["companies"] if r["symbol"]]
lost = next(r for r in names["outcome"]["rows"] if r["brand"] == "Chick-fil-a")
said = missing.check_wording(missing._question_of(names["wording"]["Chick-fil-a"]["replies"][0]), "Chick-fil-a", 400)       # the real wording passes the same check the desk makes
card = missing.card_for("Chick-fil-a", missing.reason_of(lost), kept, 1, 1, missing.Wording(text=said, source="model"), names["request"])
absent = {"seq": 0, "type": "question", "data": {"question_id": "missing0001", "kind": "missing_company", **missing_payload(card), "ts": 0}}

OUT.write_text(json.dumps({"confirm": confirm, "clarify": clarify, "approve": approve, "missing": absent}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {OUT.relative_to(REPO)}")
