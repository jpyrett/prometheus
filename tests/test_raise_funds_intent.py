"""THE PERMISSION TO RAISE FUNDS, READ FROM THE USER'S WORDS (design choices entry 142; design/direct_raise_funds_spec.md, R2; the setting, R12).

A PROGRAM decides it (`intent.raise_funds_in`, a closed list of phrases and the negations that win); the model's form has the field and only has to agree. The model's REAL replies to the spec's requests (`tests/fixtures/session_real/raise_funds_forms.json`,
captured by `scripts/capture_raise_funds_forms.py`) are given back to the real Account Manager, whose code checks every field against the words.
"""
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from prometheus.agents.account_manager import funding
from prometheus.agents.account_manager import intent as it
from prometheus.agents.account_manager.account_manager import AccountManager

FORMS = json.loads((Path(__file__).parent / "fixtures" / "session_real" / "raise_funds_forms.json").read_text(encoding="utf-8"))["requests"]


class Replay:
    """The Account Manager's model: its REAL recorded replies to the typed words, in order."""

    model_name = "openai/gpt-oss-120b"

    def __init__(self, typed):
        self.replies = list(FORMS[typed]["replies"])

    def invoke(self, messages):
        if not self.replies:
            raise AssertionError("the model was asked more times than it has recorded replies")
        return SimpleNamespace(content=self.replies.pop(0), usage_metadata={"input_tokens": 10, "output_tokens": 10})


def read(typed):
    return AccountManager(model=Replay(typed), model_id="recorded").understand_request(typed)


# ---------------------------------------------------------------- R2: the words

@pytest.mark.parametrize("typed, says", [
    ("Buy 2 shares of ANT and raise funds if needed", True),
    ("buy 5 MSFT, sell some stock to pay for it if you have to", True),
    ("Buy 2 shares of ANT, raise the money", True),
    ("free up cash to buy MU", True),
    ("free up some money for ANT", True),
    ("take profits to buy MU", True),
    ("Raise cash if you need to and buy 2 ANT", True),
    ("Buy 2 shares of ANT", False),
    ("Sell 10 shares of MSFT", False),
    ("Sell 5 shares of MSFT to buy MU", False),                                                  # the user's OWN sale: not a permission to sell something else
    ("Buy 2 shares of ANT but don't sell anything", False),
    ("Buy 2 shares of ANT, don't raise funds", False),
    ("do not raise the money, just buy ANT", False),
    ("buy ANT without selling anything", False),
    ("buy MU and raise cash if you need to, but don't sell anything", False),                      # a negation anywhere wins
    ("I want to raise funds for MU, no selling", False),
])
def test_the_closed_list_of_phrases_and_the_negations_that_win(typed, says):
    assert it.raise_funds_in(typed) is says
    assert it.denies_raise_funds(typed) is (not says and any(w in typed.lower() for w in ("don't", "do not", "without", "no selling")))


@pytest.mark.parametrize("typed, says", [("Buy 2 shares of ANT and raise funds if needed", True), ("Buy 2 shares of ANT", False), ("Buy 2 shares of ANT but don't sell anything", False), ("Buy 2 shares of ANT, don't raise funds", False),
                                          ("buy 5 MSFT, sell some stock to pay for it if you have to", True), ("Sell 10 shares of MSFT", False)])
def test_the_real_models_replies_agree_and_the_form_carries_it(typed, says):
    intent = read(typed)
    assert intent.raise_funds is says and intent.action == "order" and (" (you said to raise funds if needed" in intent.describe()) is says
    assert ("raise_funds" in intent.model_dump()) is says                                          # a dump has no `raise_funds` unless the user said it (the research cache is keyed by the dump)


def test_the_real_reply_for_a_purchase_with_two_stocks_and_one_with_dollars_keep_the_permission_on_every_order_of_the_request():
    two = read("Buy 2 shares of ANT and 3 shares of MSFT and raise funds if needed")
    assert two.raise_funds and [(o.side, o.symbol, o.shares) for o in two.orders] == [("buy", "ANT", Decimal(2)), ("buy", "MSFT", Decimal(3))]
    dollars = read("Buy $200 of ANT and raise funds if needed")
    assert dollars.raise_funds and (dollars.orders[0].dollars, dollars.orders[0].shares) == (Decimal(200), None)


def test_the_shares_typed_with_a_recommendation_keep_the_permission_too():
    one = read("buy 10 of MU if recommended and raise funds if needed")
    assert (one.action, one.shares_of, one.raise_funds) == ("buy", {"MU": Decimal(10)}, True) and one.said_how_much()
    each = read("Buy 5 shares of each of the top 3 oil stocks if recommended, and raise funds if needed")
    assert (each.action, each.shares_each, each.count, each.raise_funds) == ("buy", Decimal(5), 3, True)
    assert read("buy 10 of MU if the recommendation passes, but don't sell anything").raise_funds is False


def test_a_true_the_words_do_not_support_is_refused_and_a_false_the_words_contradict_is_overruled_by_them():
    base = json.loads(FORMS["Buy 2 shares of ANT and raise funds if needed"]["replies"][0])
    with pytest.raises(ValueError, match="raise_funds is true, but you did not say to raise funds"):
        it.parse_intent(json.dumps({**base, "raise_funds": True}), "Buy 2 shares of ANT")                       # the model made it up: refused, as a price limit that is not in the words is
    assert it.parse_intent(json.dumps({**base, "raise_funds": False}), "Buy 2 shares of ANT and raise funds if needed").raise_funds is True      # the words decide
    with pytest.raises(ValueError, match="raise_funds must be true or false"):
        it.parse_intent(json.dumps({**base, "raise_funds": "yes"}), "Buy 2 shares of ANT and raise funds if needed")
    without = {k: v for k, v in base.items() if k != "raise_funds"}                                              # an older reply with no such field: the words decide
    assert it.parse_intent(json.dumps(without), "Buy 2 shares of ANT and raise funds if needed").raise_funds is True and it.parse_intent(json.dumps(without), "Buy 2 shares of ANT").raise_funds is False


def test_a_look_is_never_a_sale_the_permission_goes_only_with_buying_and_orders():
    look = it.parse_intent(json.dumps({"action": "research", "topic": "oil", "symbols": [], "companies": [], "count": 3, "allocation": "default", "budget": None, "raise_funds": False}), "Recommend 3 oil stocks and raise funds if needed")
    assert look.action == "research" and look.raise_funds is False
    with pytest.raises(ValueError, match="holdings are sold only to pay for a purchase"):
        it.TradeIntent(request="x", action="research", topic="oil", raise_funds=True)


def test_the_prompt_tells_the_model_what_the_field_is_and_that_it_is_never_a_sell_order():
    from prometheus.prompts import account_manager_request as prompt

    assert '"raise_funds": false' in prompt.SYSTEM and "- raise_funds:" in prompt.SYSTEM and "never a trade of its own" in prompt.SYSTEM and "raise_funds, orders" in prompt.RETRY


# ---------------------------------------------------------------- R12: the setting ALWAYS_RAISE_FUNDS

@pytest.mark.parametrize("value, on", [("true", True), ("True", True), ("1", True), ("yes", True), ("on", True), ("false", False), ("0", False), ("", False), ("maybe", False)])
def test_the_setting_is_true_only_for_true_yes_on_or_1(monkeypatch, value, on):
    monkeypatch.setenv(funding.ALWAYS_SETTING, value)
    assert funding.always_raise_funds() is on
    assert funding.permission(False, "Buy 2 shares of ANT") == ("setting" if on else None)
    assert funding.permission(True, "Buy 2 shares of ANT") == "words"                                 # the words come first


def test_the_setting_is_read_through_rag_config_setting_and_never_from_the_dotenv_file_directly():
    source = Path(funding.__file__).read_text(encoding="utf-8")
    assert "from prometheus.rag.config import setting" in source and 'setting(ALWAYS_SETTING, "false")' in source and "dotenv" not in source and "os.environ" not in source


def test_a_dont_sell_anything_in_the_words_wins_over_the_setting_for_that_request_and_only_that_request(monkeypatch):
    monkeypatch.setenv(funding.ALWAYS_SETTING, "true")
    assert funding.permission(False, "Buy 2 shares of ANT but don't sell anything") is None
    assert funding.permission(False, "Buy 2 shares of ANT") == "setting"


def test_the_conftest_forces_the_setting_off_for_every_test():
    import os

    assert os.environ["ALWAYS_RAISE_FUNDS"] == "false" and funding.always_raise_funds() is False


def test_the_session_flag_sets_the_setting_for_the_session_and_puts_it_back_and_without_the_flag_nothing_is_touched(monkeypatch):
    import os

    from prometheus.application import session

    args = session.build_parser().parse_args(["--prompt", "Buy 2 shares of ANT", "--always-raise-funds"])
    assert args.always_raise_funds is True and session.build_parser().parse_args(["--prompt", "x"]).always_raise_funds is None      # the default is the setting (the environment and .env), not the flag
    with session.always_raise_funds_for(args.always_raise_funds):
        assert os.environ[funding.ALWAYS_SETTING] == "true" and funding.always_raise_funds() is True
    assert os.environ[funding.ALWAYS_SETTING] == "false" and funding.always_raise_funds() is False                              # put back as it was (the conftest sets false)
    monkeypatch.delenv(funding.ALWAYS_SETTING)
    with session.always_raise_funds_for(True):
        assert os.environ[funding.ALWAYS_SETTING] == "true"
    assert funding.ALWAYS_SETTING not in os.environ                                                                              # it was not set: removed again
    with session.always_raise_funds_for(None):
        assert funding.ALWAYS_SETTING not in os.environ
    import inspect

    assert "functools.wraps" in inspect.getsource(session.with_always_raise_funds) and "@with_always_raise_funds" in inspect.getsource(session.run_session)
