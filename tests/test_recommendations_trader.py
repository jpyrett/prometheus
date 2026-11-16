"""The recommendations run with the Trader (Trader v1 spec section 16): review, decide, TRADE, report, and the command's `--trade` flags. Everything is a fake: the Advisor is the real one over fake numbers, the Data Clerk
answers the Critic's requests and the Trader's quotes and bars from dictionaries, the critic's model is the prompt-reading ReviewerModel, and the Trader has no model. The trades ledger is in memory (the conftest guard fails
a test that writes the real one). No network, no model, no paid API."""
import json
from decimal import Decimal

import pytest
from advisor_helpers import MemoryLedger
from critic_helpers import at, profile
from test_recommendations_critic import NOW, Clerk, fake_main, go
from trader_helpers import MarketClerk, plan, weekday_bars

from prometheus.agents.account_manager.account_manager import AccountManager
from prometheus.agents.trader.trader import Trader
from prometheus.application import recommendations as rec
from prometheus.common import work_order as wo

QUOTES = {"PZZA": 100.0, "YUM": 100.0}               # the two recommendations the fake data approves: one share each, a reference price of 100, an entry zone of 98 to 102
ALL = "DPZ PZZA YUM"


def manager(clerk=None, **kw):
    """The Account Manager of the run: Jacob's profile for the user "t" (Ask mode), and a Data Clerk that answers the Trader's quotes and bars."""
    return AccountManager(model=object(), profile=profile(user="t"), clerk=clerk or MarketClerk(QUOTES), **kw)


def trade(text=ALL, trader=None, acct=None, approver=lambda card: True, **kw):
    """The run with a Trader that has no model and a ledger in memory."""
    trader = trader or Trader(use_model=False, ledger=MemoryLedger())
    result, clerk, _ = go(text, trader=trader, manager=acct or manager(), approver=approver, **kw)
    return result, trader


# ---------------------------------------------------------------- run()

def test_a_run_with_a_trader_executes_the_permitted_trades_after_the_decisions_and_the_order_is_traded():
    """Design choices entry 91: nothing limits a purchase but the cash. AMENDED (entry 106): the three approved purchases used to be sized to all of the $1,000 each (10 shares at 100) and the Trader found no cash for the third;
    with no split asked the money is spread round-robin in both modes, so DPZ is 4 shares and PZZA and YUM 3 each (by hand: three rounds of $300, and the fourth round's first share is DPZ's). DPZ has no quote (rejected), and
    PZZA and YUM both fill: nothing is rejected for cash."""
    result, trader = trade()
    order = result.order
    assert order.status == "traded" and order.schema_version == 4 and [d["will_trade"] for d in order.trade_decisions] == ["yes_user", "yes_user", "yes_user"] and [d["shares"] for d in order.trade_decisions] == [4, 3, 3]
    assert [(t["trade_id"], t["symbol"], t["side"], t["shares"], t["status"], t.get("reject_code")) for t in order.trades] == [("T1", "DPZ", "buy", 4, "rejected", "price_unavailable"), ("T2", "PZZA", "buy", 3, "filled", None),
                                                                                                                              ("T3", "YUM", "buy", 3, "filled", None)]
    assert order.trades[1]["fill"]["price"] == Decimal("100.0") and order.trades[1]["fill"]["price_source"] == "quote" and order.trader_runs[0]["mode"] == "paper" and order.trader_runs[0]["use_model"] is False and order.trades[1]["placed_by"] == "code"
    assert order.trader_runs[0]["portfolio_before"]["cash"] == Decimal("1000.0") and order.trader_runs[0]["portfolio_after"]["cash"] == Decimal("400.0")
    assert {"critic", "decide", "trade"} <= set(result.seconds) and list(result.seconds)[-1] != "relevance"


def test_without_a_trader_the_run_is_what_it_was_nothing_is_traded_and_no_quote_is_asked_for():
    clerk = MarketClerk(QUOTES)
    result, _, _ = go(ALL, manager=manager(clerk), approver=lambda card: True)
    order = result.order
    assert order.status == "decided" and order.trades == [] and order.trader_runs == [] and "trade" not in result.seconds and clerk.orders == [] and not [n for n in result.notes if "Trader" in n]
    text = rec.report(result)
    assert "TRADER" not in text and "Nothing here is bought or sold." in text and "SIMULATED" not in text


def test_the_report_has_the_trader_section_after_the_critics_and_says_the_trades_are_simulated():
    result, _ = trade()
    text = rec.report(result)
    assert text.index("CRITIC REVIEW  (model z-ai/glm-5.3") < text.index("TRADER  (no model: the program placed the orders; PAPER mode, prices: the current quotes") < text.index("MODEL CALLS (Advisor)")
    assert "T1" in text and "FILLED 3 at 100.0000" in text and "account  cash $1,000.00 -> $400.00" in text and "kept in memory only: the account book was not changed" in text
    assert "The trades above are SIMULATED (paper or backtest): no order was sent to a broker and no money moved." in text and "Nothing here is bought or sold." not in text
    assert "The Trader can only place trades the Account Manager decided on" in text and "RANKING, best to worst" in text                # a `traded` order is a finished one: the whole report is there


def test_a_decision_that_is_still_waiting_for_the_user_is_not_traded_and_the_run_says_the_trader_did_not_run():
    clerk = MarketClerk(QUOTES)
    result, trader = trade(approver=None, acct=manager(clerk))
    assert result.order.status == "reviewed" and result.order.trades == [] and result.order.trader_runs == [] and clerk.orders == []
    assert any("the Trader did not run: no decision of this order permits a trade" in n for n in result.notes) and "TRADER  (" not in rec.report(result)


def test_when_every_decision_is_a_no_the_trader_does_not_run():
    result, _ = trade(approver=lambda card: False)
    assert [d["will_trade"] for d in result.order.trade_decisions] == ["no_user_denied"] * 3 and result.order.trades == [] and result.order.status == "decided"
    assert any("the Trader did not run" in n for n in result.notes)


def test_a_trader_without_the_critic_has_nothing_to_trade_and_the_run_says_so():
    clerk = MarketClerk(QUOTES)
    trader = Trader(use_model=False, ledger=MemoryLedger())
    result, _, _ = go(ALL, critic=False, trader=trader, manager=manager(clerk))
    assert result.order.status == "recommended" and result.order.trades == [] and clerk.orders == [] and any("the Trader did not run: no recommendation was reviewed and decided on" in n for n in result.notes)


def test_the_trader_asks_for_its_quotes_through_the_account_manager_and_the_runners_data_clerk_when_no_manager_is_given():
    """With no `manager` the run builds one over ITS clerk: the quotes the Trader needs come through the same Data Clerk (here the fake that also answers the Critic), not through a clerk the Account Manager makes."""
    clerk, trader = Clerk(), Trader(use_model=False, ledger=MemoryLedger())
    from advisor_helpers import EMPTY_PORTFOLIO, TODAY, Scripted, plan_reply
    from critic_helpers import ReviewerModel
    from prometheus.agents.advisor.advisor import Advisor
    from prometheus.agents.critic.critic import Critic
    from test_recommendations import Relevance, search, theses

    advisor = Advisor(model=Scripted(plan_reply(), theses("DPZ", "PZZA", "YUM")), clerk=clerk, ledger=MemoryLedger())
    asked = []
    original = clerk.request_quotes
    clerk.request_quotes = lambda order: asked.append((order.requester, list(order.subject))) or original(order)
    called = AccountManager(model=object(), profile=profile(user="t"), clerk=clerk)                        # what `run` builds when none is given, with the profile the decisions need
    result = rec.run(rec.classify(ALL, None, search), clerk=clerk, advisor=advisor, portfolio=EMPTY_PORTFOLIO, today=TODAY, relevance_searches=(Relevance({}),), critic=Critic(model=ReviewerModel(), clerk=clerk, ledger=MemoryLedger()),
                     manager=called, now=NOW, approver=lambda card: True, trader=trader)
    assert result.order.status == "traded" and ("account_manager", ["DPZ", "PZZA", "YUM"]) in asked and called.clerk is clerk


def test_the_funding_answer_the_approver_and_book_reach_the_trader():
    asked = []
    result, trader = trade(funding=plan("PZZA", status="funded"), book=True, approver=lambda card: asked.append(card.get("kind", "decision")) or True)
    run = result.order.trader_runs[0]
    assert run["funding"]["status"] == "funded" and "already covers" in run["funding"]["note"]                                                    # the RaiseFunds answer arrived as the funding plan
    assert any("the fills were not booked" in e and "no saved account" in e for e in run["errors"]) and run["booked"] is False and asked == ["decision", "decision", "decision"]    # book was asked for; this runner holds no saved account


def test_a_backtest_enters_at_the_next_sessions_open_from_the_stored_bars_and_asks_for_no_quote():
    rows = [(100.0, 101.0, 99.0, 100.5), (100.0, 100.8, 99.2, 100.4), (100.4, 100.9, 99.9, 100.6)]
    bars = {s: weekday_bars("2026-10-23", rows) for s in ("PZZA", "YUM", "SPY")}
    clerk = MarketClerk(QUOTES, bars=bars)
    result, trader = trade(trader=Trader(mode="backtest", use_model=False, ledger=MemoryLedger()), acct=manager(clerk))
    order = result.order
    assert clerk.asked("quotes") == [] and clerk.asked("bars") == [["DPZ", "PZZA", "YUM", "SPY"]] and order.trader_runs[0]["mode"] == "backtest"
    assert [(t["symbol"], t["status"], (t["fill"] or {}).get("date"), (t["fill"] or {}).get("price_source")) for t in order.trades if t["status"] == "filled"] == [("PZZA", "filled", "2026-10-26", "bar_open"), ("YUM", "filled", "2026-10-26", "bar_open")]    # AMENDED (entry 111): the open of 100.00 is within the limit of 100.00 now, so it is the price (with 5 bps of slippage the open plus it was over the limit and the limit was the price: `bar_limit`); DPZ has no stored bar; PZZA and YUM fill (entry 106: the money is shared a share at a time, so YUM has its cash)
    assert [(t["fill"]["date"], t["fill"]["price_source"], t["outcome"]["status"]) for t in order.trades if t["status"] == "filled"] == [("2026-10-26", "bar_open", "open")] * 2          # Friday's decision enters on Monday at the open of 100.00 (AMENDED, entry 111: the open plus slippage was over the Advisor's limit of 100.00, and the limit was the price)
    assert order.status == "traded" and "outcome  T2" in rec.report(result)               # T2: the filled one


def test_a_review_made_on_another_day_than_the_run_is_expired_so_nothing_is_permitted_and_the_trader_does_not_run():
    result, _ = trade(now=at("2026-10-26 10:30"))
    assert all("record_expired" in [b["code"] for b in r["blockers"]] for r in result.order.reviews) and result.order.trades == [] and result.order.trader_runs == []
    assert any("the Trader did not run" in n for n in result.notes)


# ---------------------------------------------------------------- the command

class Recorded:
    """`Trader` replaced by a class that records how `main` built it and what it was asked, and returns the order as it is."""

    def __init__(self, monkeypatch):
        self.built, self.asked = [], []
        monkeypatch.setattr("prometheus.agents.trader.trader.Trader", self.make)

    def make(self, **kw):
        self.built.append(kw)
        outer = self

        class Fake:
            def execute_trades(self, order, account, portfolio=None, now=None, funding=None, approver=None, book=False):
                outer.asked.append({"mode": kw.get("mode"), "portfolio": portfolio, "now": now, "funding": funding, "book": book})
                return order

        return Fake()


AUTO = {"user": "Jacob Pyrett", "risk_tolerance": "moderate", "trade_mode": "ask"}


def with_profile(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(AUTO))
    return str(path)


def test_the_command_without_trade_builds_no_trader_and_asks_for_no_quote(monkeypatch, capsys):
    seen = Recorded(monkeypatch)
    code, out, _ = fake_main(monkeypatch, capsys, ALL, "--no-ledger", "--cash", "1000", "--now", "2026-10-23 10:30", "--no-ask")
    assert code == 0 and seen.built == [] and "TRADER" not in out.out and "Nothing here is bought or sold." in out.out


def test_trade_alone_is_a_paper_trader_with_the_model_and_the_real_ledger_and_the_flags_change_each(monkeypatch, capsys, tmp_path):
    seen = Recorded(monkeypatch)
    base = [ALL, "--cash", "1000", "--now", "2026-10-23 10:30", "--mode", "auto", "--profile", with_profile(tmp_path)]
    fake_main(monkeypatch, capsys, *base, "--trade")
    fake_main(monkeypatch, capsys, *base, "--trade", "backtest", "--no-trader-model")
    fake_main(monkeypatch, capsys, *base, "--trade", "paper", "--no-ledger")
    assert [(b["mode"], b["use_model"], b["ledger"].__class__.__name__) for b in seen.built] == [("paper", True, "NoneType"), ("backtest", False, "NoneType"), ("paper", True, "NullLedger")]
    assert len(seen.asked) == 3 and all(a["book"] is False and a["funding"] is None for a in seen.asked)             # this runner has no saved account and no funds node: nothing is booked, no plan is made
    assert seen.asked[0]["portfolio"].user == "Jacob Pyrett" and seen.asked[0]["portfolio"].cash_avail == Decimal("1000.0") and seen.asked[0]["now"] == at("2026-10-23 10:30")


def test_the_command_prints_the_trader_section_and_saves_the_order_with_its_trades(monkeypatch, capsys, tmp_path):
    save = tmp_path / "order.json"
    code, out, _ = fake_main(monkeypatch, capsys, ALL, "--cash", "1000", "--now", "2026-10-23 10:30", "--mode", "auto", "--profile", with_profile(tmp_path), "--trade", "--no-trader-model", "--no-ledger", "--save", str(save))
    assert code == 0 and "TRADER  (no model: the program placed the orders; PAPER mode" in out.out and "FILLED" in out.out and "The trades above are SIMULATED" in out.out and "(work order saved to" in out.out
    saved = wo.WorkOrder.model_validate_json(save.read_text())
    assert saved.status == "traded" and saved.schema_version == 4 and saved.trades and {t["status"] for t in saved.trades} == {"filled"} and saved.trader_runs[0]["booked"] is False


def test_a_command_run_that_traded_nothing_says_so_and_still_finishes_with_exit_0(monkeypatch, capsys):
    code, out, _ = fake_main(monkeypatch, capsys, ALL, "--no-ledger", "--cash", "1000", "--now", "2026-10-23 10:30", "--no-ask", "--trade", "--no-trader-model")
    assert code == 0 and "the Trader did not run: no decision of this order permits a trade" in out.out and "TRADER  (" not in out.out


@pytest.mark.parametrize("args, message", [(["--no-critic", "--trade"], "--trade needs the Critic"), (["--no-critic", "--trade", "backtest"], "--trade needs the Critic"), (["--no-trader-model"], "--no-trader-model goes with --trade"),
                                           (["--no-critic", "--no-trader-model"], "--no-trader-model goes with --trade"), (["--trade", "live"], "invalid choice")])
def test_the_trader_flags_that_do_not_go_together_are_refused_before_anything_is_paid_for(args, message, capsys):
    with pytest.raises(SystemExit) as stop:
        rec.main(["DPZ", *args])
    assert stop.value.code == 2 and message in capsys.readouterr().err
