"""The TRADER section of a report (prometheus/agents/trader/report.py, Trader spec section 16): plain text from the records, no model. Every line is checked against a run made over fakes."""
import copy

from test_trader import BARS, PLACE_ALL, funded_run, go, last
from trader_helpers import ScriptedModel, account, answer, call, decided, row, say, synthetic

from prometheus.agents.trader.report import account_line, header, outcome_line, trade_line, trade_section
from prometheus.common import work_order as wo


def section(order):
    return "\n".join(trade_section(order))


def test_a_run_without_a_model_says_so_and_a_model_that_could_not_start_says_that():
    order, *_ = go(decided(), {"PG": 167.0})
    assert "no model: the program placed the orders" in header(order)
    order, *_ = go(decided(), {"PG": 167.0}, use_model=True)
    assert "no model could run: the program placed the orders" in header(order) and any(l.startswith("  trader note: the Trader's model could not start") for l in trade_section(order))


def test_a_resting_order_shows_active_and_why():
    order, *_ = go(decided(), {"PG": 190.0})
    assert trade_section(order)[1] == "  T1     buy  44 of 44 PG     limit 167.10  ACTIVE: resting: the quote 190.0000 is above the limit 167.1000   placed by code"


def test_the_funding_sells_are_listed_under_their_purchase_with_the_answer_to_the_question_and_the_positions_that_changed():
    order, cards, acct = funded_run()
    lines = trade_section(order)
    assert lines[1].startswith("  T1     buy  20 of 20 PG") and lines[2].lstrip().startswith("T1-F1  sell 5 of 5 MSFT") and lines[2].endswith("(funds T1)") and lines[3].lstrip().startswith("T1-F2  sell 2 of 2 TSLA")
    assert "  funding  RaiseFunds said plan: the user approved the funding sells (you said yes)" in lines
    # AMENDED (entry 111): the cash was $808.85 with 5 bps of slippage on every fill; with none it is $811.23
    assert any(l.startswith("  account  cash $51.23 -> $811.23; MSFT 50 -> 45 shares; TSLA 20 -> 18 shares; PG 0 -> 20 shares; kept in memory") for l in lines)


def test_a_no_to_the_funding_question_is_shown_as_the_users_no():
    order, cards, acct = funded_run(approver=lambda card: False)
    assert any(l.startswith("  funding  RaiseFunds said plan: the funding sells were not approved: the user said no (you said no)") for l in trade_section(order))


def test_a_trades_note_is_shown_under_it_in_the_models_words():
    model = ScriptedModel([call("place_order", trade_id="T1", note="inside the range"), call("notify_account_manager"), say(answer("ok", ("T1", "filled")))])
    order, *_ = go(decided(), {"PG": 167.0}, model=model)
    lines = trade_section(order)
    assert lines[1].startswith("  T1") and lines[2] == "      note: inside the range"


def test_an_open_backtest_position_shows_the_last_close_and_the_unrealised_gain_and_a_stop_shows_a_loss():
    bars = {"PG": [b for b in BARS["PG"][:1]] + [dict(BARS["PG"][1])] + [dict(BARS["PG"][2], high=167.0, low=166.0, close=166.5)], "SPY": BARS["SPY"]}
    order, *_ = go(decided(), bars=bars, mode="backtest")
    # AMENDED (entry 111): the entry was the open plus 5 bps (unrealised +$18.35, +0.25%, excess -0.75%); it is the open itself now
    assert "  outcome  T1  still open on 2026-11-09 at 166.5000 after 1 session: unrealised +$22.00 (+0.30%), benchmark +1.00% (excess -0.70%)" in trade_section(order)
    stopped = {"PG": [BARS["PG"][0], BARS["PG"][1], dict(BARS["PG"][2], open=160.0, high=161.0, low=150.0, close=152.0)], "SPY": BARS["SPY"]}
    order, *_ = go(decided(), bars=stopped, mode="backtest")
    line = next(l for l in trade_section(order) if "outcome" in l)
    assert "closed by stop on 2026-11-09" in line and "gain -$" in line and "(-" in line


def test_an_outcome_with_no_later_bar_says_so():
    assert outcome_line({"trade_id": "T1", "outcome": {"status": "open"}}) == "  outcome  T1  open: no bar after the entry is stored" and outcome_line({"trade_id": "T1", "outcome": None}) is None


def test_a_backtest_with_no_bars_stored_shows_the_trade_as_waiting_and_no_outcome():
    order, *_ = go(decided(), bars={}, mode="backtest")
    lines = trade_section(order)
    assert "ACTIVE: waiting for the next session" in lines[1] and not any("outcome" in l for l in lines)


def test_a_refused_run_shows_its_header_and_its_reason_and_not_a_claim_that_there_were_no_decisions():
    order, *_ = go(decided(), {"PG": 167.0}, mode="live")
    lines = trade_section(order)
    assert lines[0].startswith("TRADER  (no model: the program placed the orders; LIVE mode, prices: no prices;") and lines[1] == "  the Trader did not run: nothing was traded (the reason is below)"
    assert lines[2].startswith("  trader note: the Trader's mode must be paper or backtest") and len(lines) == 3


def test_an_order_with_no_permitted_decision_says_so_and_stays_decided_not_traded():
    out, acct, *_ = go(decided(answer=False), {"PG": 167.0})
    lines = trade_section(out)
    assert out.trades == [] and out.status == "decided" and acct.clerk.orders == [] and "  no trades: no decision of this order permits one" in lines
    assert lines[-1] == "  summary (by code): Paper run: no trades to place. Cash $10,000.00 to $10,000.00."


def test_the_line_of_a_trade_that_was_never_placed_says_so():
    assert trade_line({"trade_id": "T1", "side": "buy", "symbol": "PG", "shares": 5, "shares_approved": 5, "status": "new", "order_type": "market"}).strip().startswith("T1     buy  5 of 5 PG") and "NOT PLACED" in trade_line(
        {"trade_id": "T1", "side": "buy", "symbol": "PG", "shares": 5, "shares_approved": 5, "status": "new", "order_type": "market"})


def test_an_account_line_needs_a_before_and_an_after():
    assert account_line({}) is None and account_line({"portfolio_before": {"cash": 1.0, "positions": []}}) is None
