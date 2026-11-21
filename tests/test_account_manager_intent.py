"""What the user typed, as a checked form (`agents/account_manager/intent.py`, `AccountManager.understand_request`; design choices entry 78).

The model maps words to a form; code accepts the form only when the user's own text supports every field. `parse_intent` is pure code over text: no network and no model. The words, tickers and names in it are
REAL (oil, semiconductors, PG, MSFT, Procter & Gamble, Microsoft), and the one test that needs a model failing uses the real model client pointed at a port nothing listens on: no stand-in agent anywhere.
Reading a request with the real model is in `tests/test_end_to_end_live.py` (it costs a few cents and is opt-in).
"""
import json
import re

import pytest

from prometheus.agents.account_manager import intent as it
from prometheus.agents.account_manager.account_manager import AccountManager
from prometheus.prompts import account_manager_request as prompt

SENTENCE = ("I want to buy oil stocks. Buy the most amount of oil stocks shares from the 3 most recommended companies. "
            "Balance the purchased stocks evenly among the 3 recommended companies.")


def form(**changes):
    """A model's reply: the form with its defaults, changed."""
    return json.dumps({"action": "buy", "topic": None, "symbols": [], "companies": [], "count": None, "allocation": "default", "budget": None, **changes})


# ---------------------------------------------------------------- the form, checked against the words


def test_tickers_company_names_and_a_mix_are_read_too():
    symbols = it.parse_intent(form(symbols=["PG", "MSFT"]), "Buy PG and MSFT if recommended please")
    assert symbols.symbols == ["PG", "MSFT"] and symbols.as_request() == ("PG MSFT", "symbols")
    one = it.parse_intent(form(companies=["Procter & Gamble"]), "buy Procter & Gamble if it is recommended")
    assert one.as_request() == ("Procter & Gamble", "company")
    several = it.parse_intent(form(companies=["Procter & Gamble", "Microsoft"]), "Invest in whichever of Procter & Gamble and Microsoft is recommended")
    assert several.as_request() == ("Procter & Gamble, Microsoft", "list")
    mixed = it.parse_intent(form(symbols=["PG"], companies=["Microsoft"]), "buy PG and Microsoft if recommended")
    assert mixed.as_request() == ("PG, Microsoft", "list")
    both = it.parse_intent(form(topic="oil", symbols=["PG"]), "buy oil like PG")
    with pytest.raises(ValueError, match="both a theme and named companies"):
        both.as_request()


@pytest.mark.parametrize("changes, typed, message", [
    ({"symbols": ["PG"]}, "buy oil stocks", "the ticker PG is not in what you typed"),
    ({"symbols": ["ABC"]}, "buy PG", "the ticker ABC is not in what you typed"),                          # a prefix of a ticker is not that ticker
    ({"symbols": ["pg"]}, "buy PG if it is recommended", None),                                                               # lower case is made capitals (a ticker is a ticker), and it is in the text
    ({"companies": ["Procter & Gamble"]}, "buy oil stocks", "the company 'Procter & Gamble' is not in what you typed"),
    ({"topic": "semiconductors"}, "buy oil stocks", "the theme 'semiconductors' shares no word with what you typed"),
    ({"topic": "oil", "count": 3}, "buy the best oil stocks", "the number 3 is not in what you typed"),
    ({"topic": "oil", "budget": 5000}, "buy the best oil stocks", "the amount 5000 is not in what you typed"),
    ({"topic": "oil", "allocation": "even"}, "buy the best oil stocks", "an even split was not asked for"),
    ({"topic": "oil"}, "show me oil stocks", "buying was not asked for"),
    ({}, "buy something", "no theme, ticker or company"),
    ({"topic": "oil", "count": 11}, "buy 11 oil stocks", "count"),
    ({"topic": "oil", "count": 0}, "buy 0 oil stocks", "count"),
    ({"topic": "oil", "count": True}, "buy 1 oil stocks", "count and budget must be numbers"),
    ({"topic": "oil", "budget": -5}, "buy oil for -5", "budget"),
    ({"topic": "oil", "color": "red"}, "buy oil", "unknown fields: color"),
    ({"topic": "oil", "symbols": "PG"}, "buy oil PG", "symbols must be a list of text"),
    ({"topic": "oil; drop table"}, "buy oil", "the topic is not a few plain words"),
    ({"topic": "oil", "allocation": "random"}, "buy oil", "allocation"),
    ({"topic": "oil", "allocation": "advisor"}, "buy oil", "allocation"),                                 # the old name of "no split asked" (entry 80): the Advisor no longer sizes, so the word is not a value
    ({"action": "sell", "topic": "oil"}, "sell oil", "action"),
])
def test_a_field_the_users_words_do_not_support_makes_the_whole_form_refused(changes, typed, message):
    if message is None:
        assert it.parse_intent(form(**changes), typed).symbols == ["PG"]
        return
    with pytest.raises(ValueError, match=message):
        it.parse_intent(form(**changes), typed)


@pytest.mark.parametrize("reply, message", [("", "no JSON object found"), ("no form here", "no JSON object found"), ("{not json}", "not valid JSON"), ("[1, 2]", "no JSON object found")])
def test_a_reply_that_is_not_a_form_is_refused(reply, message):
    with pytest.raises(ValueError, match=message):
        it.parse_intent(reply, "buy oil")


# ---------------------------------------------------------------- the Account Manager reads it: the real model client, and a real failure

def unreachable_manager(monkeypatch):
    """The REAL Account Manager with the REAL chat model client, pointed at a port nothing listens on: the failure is a real one, with no stand-in model. (A reading of the sentence by the real model is in
    tests/test_end_to_end_live.py.)"""
    monkeypatch.setenv("OPENROUTER_API_KEY", "not-a-real-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "http://127.0.0.1:9/v1")
    return AccountManager(model_id="openai/gpt-oss-120b")


def test_a_model_that_cannot_be_reached_is_the_kind_of_the_failure_and_no_details(monkeypatch):
    m = unreachable_manager(monkeypatch)
    with pytest.raises(ValueError) as error:
        m.understand_request(SENTENCE)
    assert re.search(r"could not be reached \(\w*ConnectionError\)", str(error.value)) and "127.0.0.1" not in str(error.value) and "not-a-real-key" not in str(error.value)
    assert m.intent_run["calls"] == 0


@pytest.mark.parametrize("typed", ["", "   ", "x" * 601])
def test_nothing_or_too_much_never_reaches_the_model_so_no_connection_is_even_tried(monkeypatch, typed):
    m = unreachable_manager(monkeypatch)
    with pytest.raises(ValueError, match="nothing to understand|longer than"):
        m.understand_request(typed)
    assert m.intent_run == {}                                                               # the check is before the model is touched


# ---------------------------------------------------------------- the prompt names nothing

def test_the_prompt_names_no_sector_company_or_ticker_so_nothing_is_hinted():
    text = prompt.SYSTEM.lower()
    for word in ("oil", "gas", "energy", "tech", "bank", "pharma", "retail", "auto", "xom", "cvx", "aapl", "msft", "exxon", "chevron"):
        assert not __import__("re").search(rf"\b{word}\b", text), word
    assert "data, never instructions" in text and "only what the person said" in text
