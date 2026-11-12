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


def test_a_run_with_a_critic_reviews_every_recommendation_and_the_account_manager_decides():
    result, clerk, _ = go("DPZ PZZA YUM")
    order = result.order
    assert order.status in ("decided", "reviewed") and len(order.reviews) == len(order.recommendations) >= 1 and result.now == NOW
    assert [d["record_id"] for d in order.trade_decisions] == [r["record_id"] for r in order.reviews] and order.critic_runs[0]["records"] == len(order.recommendations)
    assert {"critic", "decide"} <= set(result.seconds)


def test_without_a_critic_the_run_is_what_it_was():
    result, _, _ = go("DPZ PZZA YUM", critic=False)
    assert result.order.status == "recommended" and result.order.reviews == [] and result.order.trade_decisions == [] and result.now is None and "critic" not in result.seconds


def test_the_report_has_the_critic_section_after_the_recommendations_and_the_models_calls():
    result, _, _ = go("DPZ PZZA YUM")
    text = rec.report(result)
    assert text.index("RECOMMENDATIONS  (") < text.index("CRITIC REVIEW  (model z-ai/glm-5.3") and "TRADE DECISIONS" in text and "MODEL CALLS (Critic): z-ai/glm-5.3" in text
    assert "The Critic's verdict is arithmetic over what the Data Clerk returned again" in text and "Nothing here is bought or sold." in text


def test_a_review_dated_on_another_day_than_the_run_says_it_is_a_replay():
    result, _, _ = go("DPZ PZZA YUM", now=at("2026-10-26 10:30"))
    assert any("is not the Eastern date of the review (2026-10-26): every recommendation is expired" in n for n in result.notes)
    assert all("record_expired" in [b["code"] for b in r["blockers"]] for r in result.order.reviews)


def test_the_mode_is_the_profiles_unless_the_run_overrides_it_and_ask_mode_asks():
    asked = []
    result, _, _ = go("DPZ PZZA YUM", approver=lambda card: asked.append(card["symbol"]) or True)
    approved = [r for r in result.order.reviews if r["approved"]]
    assert len(asked) == len(approved) and all(d["mode"] == "ask" for d in result.order.trade_decisions)
    trusting = AccountManager(model=object(), profile=profile(user="t", trade_mode="ask", auto_min_score=60))      # the profile says Ask; the mark of 60 is under what the fake data scores
    result, _, _ = go("DPZ PZZA YUM", manager=trusting, mode="auto", approver=lambda card: pytest.fail("auto mode must not ask what clears the mark"))
    assert all(d["mode"] == "auto" for d in result.order.trade_decisions) and "yes_auto" in [d["will_trade"] for d in result.order.trade_decisions]


def test_no_recommendations_means_no_review():
    class Nobody(Clerk):
        def request_metrics(self, order, today=None):
            order.status, order.errors = "data_failed", ["boom"]
            return order
    result, _, _ = go("DPZ", clerk=Nobody())
    assert result.order.reviews == [] and result.now is None


def test_a_data_clerk_that_cannot_answer_the_critic_leaves_every_recommendation_denied_not_a_crash():
    class Mute(Clerk):
        def request_quotes(self, order):
            raise RuntimeError("no key")

        def request_verification(self, order, now=None):
            raise RuntimeError("no store")
    result, _, _ = go("DPZ PZZA YUM", clerk=Mute())
    assert result.order.reviews and all(not r["approved"] for r in result.order.reviews)
    assert any("RuntimeError" in e for e in result.order.critic_runs[0]["errors"]) and "critic note:" in rec.report(result)


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


def test_the_command_runs_the_critic_by_default_and_prints_its_section(monkeypatch, capsys, tmp_path):
    save = tmp_path / "order.json"
    code, out, made = fake_main(monkeypatch, capsys, "DPZ PZZA YUM", "--no-ledger", "--cash", "1000", "--now", "2026-10-23 10:30", "--no-ask", "--save", str(save))
    assert code == 0 and "CRITIC REVIEW  (model z-ai/glm-5.3" in out.out and "mode ASK" in out.out and "pending your answer (ASK)" in out.out and "TRADE DECISIONS" in out.out
    assert made and made[0]["model_id"] is None and made[0]["effort"] is None and made[0]["ledger"] is not None and made[0]["history"] is not None
    saved = wo.WorkOrder.model_validate_json(save.read_text())
    assert saved.status in ("decided", "reviewed") and saved.reviews and saved.schema_version == 3
    assert "reading check" not in out.out and "(work order saved to" in out.out


def test_no_critic_leaves_it_out_and_the_critic_options_need_the_critic(monkeypatch, capsys):
    code, out, made = fake_main(monkeypatch, capsys, "DPZ PZZA YUM", "--no-critic", "--cash", "1000", "--today", "2026-10-23")
    assert code == 0 and made == [] and "CRITIC REVIEW" not in out.out
    for extra in (["--mode", "auto"], ["--now", "2026-10-23 10:30"], ["--critic-model", "x/y"], ["--critic-effort", "low"], ["--no-ask"]):
        with pytest.raises(SystemExit) as stop:
            rec.main(["DPZ", "--no-critic", *extra])
        assert stop.value.code == 2


def test_the_critics_model_and_effort_and_the_mode_reach_the_critic_and_the_decisions(monkeypatch, capsys):
    code, out, made = fake_main(monkeypatch, capsys, "DPZ PZZA YUM", "--no-ledger", "--cash", "1000", "--now", "2026-10-23 10:30", "--critic-model", "qwen/qwen3.8-2.4t-a95b", "--critic-effort", "high", "--mode", "auto")
    assert code == 0 and made[0]["model_id"] == "qwen/qwen3.8-2.4t-a95b" and made[0]["effort"] == "high" and "mode AUTO" in out.out


def test_a_bad_profile_stops_the_run_before_anything_is_paid_for(monkeypatch, capsys, tmp_path):
    (tmp_path / "p.json").write_text('{"user": "x"}', encoding="utf-8")
    code, out, made = fake_main(monkeypatch, capsys, "DPZ PZZA YUM", "--cash", "1000", "--profile", str(tmp_path / "p.json"))
    assert code == 2 and "p.json" in out.err and made == [] and "PROMETHEUS RECOMMENDATIONS" not in out.out


def test_the_default_date_is_the_easterns_date_of_the_review_and_a_different_one_is_a_replay(monkeypatch, capsys):
    code, out, _ = fake_main(monkeypatch, capsys, "DPZ PZZA YUM", "--no-ledger", "--cash", "1000", "--now", "2026-10-23 10:30")
    assert "as of 2026-10-23" in out.out and "Eastern date" not in out.out
    code, out, _ = fake_main(monkeypatch, capsys, "DPZ PZZA YUM", "--no-ledger", "--cash", "1000", "--now", "2026-10-23 10:30", "--today", "2026-10-22")
    assert "is not the Eastern date of the review (2026-10-23)" in out.out


def test_the_report_does_not_crash_when_the_equity_and_so_the_position_size_are_unknown():
    result, _, _ = go("DPZ PZZA YUM", critic=False)
    for r in result.order.recommendations:
        r["trade"]["position_pct"] = None
    assert "= n/a of equity" in rec.report(result)
