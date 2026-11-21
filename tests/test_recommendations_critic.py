"""The recommendations run with the Critic (Critic spec 13.2): review, decide, report, and the command's new flags. Everything is a fake: the Advisor is the real one over fake numbers, the Data Clerk
answers the Critic's three requests from the same data, and the critic's model is the prompt-reading ReviewerModel."""
import json

import pytest
from advisor_helpers import EMPTY_PORTFOLIO, TODAY, MemoryLedger, Scripted, plan_reply
from critic_helpers import FakeClerk as CriticClerk, ReviewerModel, at, profile
from test_recommendations import Relevance, search, theses
from test_recommendations import Clerk as AdvisorClerk

from prometheus.agents.account_manager.account_manager import AccountManager
from prometheus.agents.advisor.advisor import Advisor
from prometheus.agents.critic.critic import Critic
from prometheus.application import recommendations as rec
from prometheus.common import work_order as wo

NOW = at("2026-10-23 10:30")       # TODAY is Friday 2026-10-23: a trading day, in the regular session


class Clerk(AdvisorClerk):
    """The Advisor's fake Data Clerk, which also answers the Critic: the quote at each last close, the sources of the order it made, every symbol listed."""
    critic_side = None

    def request_data(self, order, today=None):
        out = super().request_data(order, today)
        Clerk.critic_side = None
        self.order = out
        return out

    def _side(self):
        return CriticClerk(self.order, now=NOW, metrics=self.order.data.metrics)

    def request_metrics(self, order, today=None):
        return self._side().request_metrics(order, today) if order.requester == "critic" else super().request_metrics(order, today)      # the Critic's numbers include its own (the six-month average)

    def request_quotes(self, order):
        return self._side().request_quotes(order)

    def request_verification(self, order, now=None):
        return self._side().request_verification(order, now)


def go(request_text="DPZ", critic=True, manager=None, approver=None, now=NOW, today=TODAY, mode=None, clerk=None, **kw):
    clerk = clerk or Clerk()
    advisor = Advisor(model=Scripted(plan_reply(), theses("DPZ", "PZZA", "YUM")), clerk=clerk, ledger=MemoryLedger())
    reviewer = Critic(model=ReviewerModel(), clerk=clerk, ledger=MemoryLedger()) if critic else None
    result = rec.run(rec.classify(request_text, None, search), clerk=clerk, advisor=advisor, portfolio=EMPTY_PORTFOLIO, today=today, relevance_searches=(Relevance({}),), critic=reviewer,
                     manager=manager or AccountManager(model=object(), profile=profile(user="t")), now=now, approver=approver, mode=mode, **kw)
    return result, clerk, reviewer


# ---------------------------------------------------------------- the command

def fake_main(monkeypatch, capsys, *argv):
    clerk = Clerk()
    monkeypatch.setattr("prometheus.agents.data_clerk.data_clerk_agent.DataClerk", lambda **kw: clerk)
    monkeypatch.setattr("prometheus.agents.advisor.advisor.Advisor", lambda **kw: Advisor(model=Scripted(plan_reply(), theses("DPZ", "PZZA", "YUM")), clerk=clerk, ledger=MemoryLedger()))
    made = []
    monkeypatch.setattr("prometheus.agents.critic.critic.Critic", lambda **kw: made.append(kw) or Critic(model=ReviewerModel(), clerk=clerk, ledger=MemoryLedger(), history=kw.get("history")))
    monkeypatch.setattr(rec, "classify", lambda text, kind=None, search=None, _real=rec.classify: _real(text, kind, search or __import__("test_recommendations").search))
    monkeypatch.setattr(rec, "phrase_relevance", lambda phrase, symbols, searches=None: {s: 0.5 for s in symbols})
    code = rec.main(list(argv))
    return code, capsys.readouterr(), made


def test_no_critic_leaves_it_out_and_the_critic_options_need_the_critic(monkeypatch, capsys):
    code, out, made = fake_main(monkeypatch, capsys, "DPZ PZZA YUM", "--no-critic", "--cash", "1000", "--today", "2026-10-23")
    assert code == 0 and made == [] and "CRITIC REVIEW" not in out.out
    for extra in (["--mode", "auto"], ["--now", "2026-10-23 10:30"], ["--critic-model", "x/y"], ["--critic-effort", "low"], ["--no-ask"]):
        with pytest.raises(SystemExit) as stop:
            rec.main(["DPZ", "--no-critic", *extra])
        assert stop.value.code == 2


def test_a_bad_profile_stops_the_run_before_anything_is_paid_for(monkeypatch, capsys, tmp_path):
    (tmp_path / "p.json").write_text('{"user": "x"}', encoding="utf-8")
    code, out, made = fake_main(monkeypatch, capsys, "DPZ PZZA YUM", "--cash", "1000", "--profile", str(tmp_path / "p.json"))
    assert code == 2 and "p.json" in out.err and made == [] and "PROMETHEUS RECOMMENDATIONS" not in out.out


