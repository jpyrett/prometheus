"""The recommendations run with the Trader (Trader v1 spec section 16): review, decide, TRADE, report, and the command's `--trade` flags. Everything is a fake: the Advisor is the real one over fake numbers, the Data Clerk
answers the Critic's requests and the Trader's quotes and bars from dictionaries, the critic's model is the prompt-reading ReviewerModel, and the Trader has no model. The trades ledger is in memory (the conftest guard fails
a test that writes the real one). No network, no model, no paid API."""
import json

import pytest
from advisor_helpers import MemoryLedger
from critic_helpers import at, profile
from test_recommendations_critic import Clerk, fake_main, go
from trader_helpers import MarketClerk

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


def test_a_review_made_on_another_day_than_the_run_is_expired_so_nothing_is_permitted_and_the_trader_does_not_run():
    result, _ = trade(now=at("2026-10-26 10:30"))
    assert all("record_expired" in [b["code"] for b in r["blockers"]] for r in result.order.reviews) and result.order.trades == [] and result.order.trader_runs == []
    assert any("the Trader did not run" in n for n in result.notes)


# ---------------------------------------------------------------- the command


AUTO = {"user": "Jacob Pyrett", "risk_tolerance": "moderate", "trade_mode": "ask"}


def with_profile(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(AUTO))
    return str(path)


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
