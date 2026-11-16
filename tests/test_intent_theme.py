"""A THEME MADE OF SHORT WORDS ("AI stocks", "EV stocks", "5G companies") and a theme in quotes (design choices entry 113).

The Account Manager's prompt tells the model to leave the word "stocks" out of the topic, so "the best AI stocks" is the topic `AI`, and the old check ("the theme shares a word of three letters or more with what
you typed") could never pass for it: Jacob's own sentence was refused with "the theme 'AI' shares no word with what you typed". Two rules (T1 and T2, `agents/account_manager/intent.py`):
  T1  a theme with NO word of three letters is checked by ALL its words, each one a whole word of what was typed;
  T2  a phrase in double quotes is the theme, word for word, and that is what the Data Clerk is asked for ("AI stocks").

REAL DATA ONLY: the model's replies are its REAL ones, recorded on 2026-11-16 with the Account Manager's own prompt (`tests/fixtures/session_real/theme_request_forms.json`: one call each, nothing edited), given
back to the real Account Manager, whose code checks every field against the user's words. A refusal is a real reply read against a different typed sentence; a form for a direct order, a ticker or a company is the
real reply of `direct_order_forms.json` read against the same sentence with the names put in quotes. The live Auto session of Jacob's sentence is `auto_ai_quoted_theme.json` (see the last test).
"""
import inspect
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from prometheus.agents.account_manager import intent as it
from prometheus.agents.account_manager.account_manager import AccountManager
from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
from prometheus.application import recommendations
from prometheus.common import work_order as wo
from prometheus.common.money import money

SESSIONS = Path(__file__).parent / "fixtures" / "session_real"
THEMES = json.loads((SESSIONS / "theme_request_forms.json").read_text(encoding="utf-8"))["requests"]
ORDERS = json.loads((SESSIONS / "direct_order_forms.json").read_text(encoding="utf-8"))["requests"]
JACOB = 'I want to buy the best "AI stocks" as a theme. Buy $20000 worth of the top 3'
NO_QUOTES = "I want to buy the best AI stocks. Buy $20000 worth of the top 3"
EV = "Buy the 3 most recommended EV stocks with $5,000"
FIVE_G = "Look at the best 5G companies"
AFTER_THE_THEME = " as a theme. Buy $20000 worth of the top 3"      # what follows the theme in Jacob's sentence


def reply(typed: str) -> str:
    """The real model's one reply to this request."""
    return THEMES[typed]["replies"][0]


def read(typed: str, said: str | None = None):
    """The real reply to `typed`, checked against `said` (the sentence the user typed, which is `typed` unless a test reads the reply against another)."""
    return it.parse_intent(reply(typed), said or typed)


class Replay:
    """The Account Manager's model: its REAL recorded replies, given back."""

    model_name = "openai/gpt-oss-120b"

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def invoke(self, messages):
        self.seen.append(messages)
        return SimpleNamespace(content=self.replies.pop(0), usage_metadata={"input_tokens": 10, "output_tokens": 10})


# ---------------------------------------------------------------- the real replies

def test_the_real_model_leaves_stocks_out_of_the_theme_so_it_is_a_word_of_two_letters():
    assert {typed: json.loads(reply(typed))["topic"] for typed in THEMES} == {JACOB: "AI", NO_QUOTES: "AI", EV: "EV", FIVE_G: "5G"} and all(len(THEMES[t]["replies"]) == 1 for t in THEMES)      # the model's own words, one call each


def test_jacobs_sentence_has_the_theme_in_quotes_word_for_word_and_the_data_clerk_is_asked_for_it():
    model = Replay(THEMES[JACOB]["replies"])
    intent = AccountManager(model=model, model_id="recorded").understand_request(JACOB)
    assert (intent.action, intent.topic, intent.count, intent.budget, intent.allocation, intent.symbols, intent.companies, intent.orders) == ("buy", "AI stocks", 3, Decimal("20000"), "default", [], [], []) and len(model.seen) == 1
    assert intent.as_request() == ("AI stocks", "phrase") and intent.what() == "'AI stocks'" and intent.describe() == "buy the 3 most recommended 'AI stocks'; spending at most $20,000.00"      # not 'AI stocks' companies
    asked = recommendations.classify(*intent.as_request())
    assert (asked.kind, asked.phrase) == ("phrase", "AI stocks") and it.CLERK_PHRASE.fullmatch(asked.phrase)


def test_the_phrase_rule_is_the_data_clerks_own():
    assert it.CLERK_PHRASE_CHARS in inspect.getsource(DataClerk.find_companies) and "len(text) > 60" in inspect.getsource(DataClerk.find_companies)


@pytest.mark.parametrize("typed, topic, action, count, budget, how", [
    (NO_QUOTES, "AI", "buy", 3, Decimal("20000"), "buy the 3 most recommended 'AI' companies; spending at most $20,000.00"),
    (EV, "EV", "buy", 3, Decimal("5000"), "buy the 3 most recommended 'EV' companies; spending at most $5,000.00"),
    (FIVE_G, "5G", "research", None, None, "look at (nothing will be bought) the most recommended '5G' companies"),
])
def test_without_quotes_the_theme_is_the_models_own_short_word_and_it_is_in_the_users_words(typed, topic, action, count, budget, how):
    model = Replay(THEMES[typed]["replies"])
    intent = AccountManager(model=model, model_id="recorded").understand_request(typed)
    assert (intent.action, intent.topic, intent.count, intent.budget) == (action, topic, count, budget) and intent.as_request() == (topic, "phrase") and intent.describe() == how and len(model.seen) == 1


# ---------------------------------------------------------------- T1: a short theme is checked by its own words

@pytest.mark.parametrize("typed, said", [
    (NO_QUOTES, "buy $20000 worth of the top 3 oil stocks"),                      # AI is nowhere in the words
    (EV, "Buy the 3 most recommended EVENT stocks with $5,000"),                  # EV starts a word and is not one
    (FIVE_G, "Look at the best 5GHz companies"),                                  # 5G starts a word and is not one
])
def test_a_short_theme_whose_words_are_not_in_what_the_user_typed_is_still_refused(typed, said):
    with pytest.raises(ValueError, match=f"the theme {json.loads(reply(typed))['topic']!r} shares no word with what you typed"):
        read(typed, said)


# ---------------------------------------------------------------- T2: a phrase in quotes is the theme, word for word

@pytest.mark.parametrize("said, theme", [
    (JACOB, "AI stocks"),
    ('I want to buy the best \u201cAI stocks\u201d' + AFTER_THE_THEME, "AI stocks"),                    # curly quotes
    ('I want to buy the best "ai   stocks"' + AFTER_THE_THEME, "ai stocks"),                           # as typed, the spaces collapsed
    ('I want to buy the best "AI" stocks' + AFTER_THE_THEME, "AI"),                                    # a short phrase is a whole word of the words
    ('I want to buy the best "AI stocks" and "chips"' + AFTER_THE_THEME, "AI stocks"),                # another phrase holds nothing of the topic
])
def test_the_quoted_phrase_that_holds_the_models_topic_becomes_the_theme(said, theme):
    assert read(JACOB, said).topic == theme


@pytest.mark.parametrize("said, theme, why", [
    ('I want to buy the best AI stocks as a "theme". Buy $20000 worth of the top 3', "AI", "the quoted word has no word of the topic"),
    ('I want "AI stocks" or "AI chips". Buy $20000 worth of the top 3', "AI", "two different phrases hold it: no answer"),
    ('I want to buy the best "AI/ML stocks"' + AFTER_THE_THEME, "AI", "a slash passes the topic's check and not the Data Clerk's rule"),
    ('I want to buy the best "AI stocks that run data centers and sell chips and software to every business"' + AFTER_THE_THEME, "AI", "more than 60 characters"),
    ('I want to buy the best "AI stocks; drop table"' + AFTER_THE_THEME, "AI", "a semicolon is not in the phrase rule"),
])
def test_a_quoted_phrase_that_is_not_the_theme_or_not_a_phrase_the_clerk_takes_leaves_the_models_topic(said, theme, why):
    assert read(JACOB, said).topic == theme, why


def test_a_phrase_in_quotes_that_holds_none_of_the_topic_cannot_make_a_theme_the_words_do_not_support():
    with pytest.raises(ValueError, match="the theme 'AI' shares no word with what you typed"):
        read(JACOB, 'I want to buy the best "chip stocks" as a theme. Buy $20000 worth of the top 3')


def test_t2_applies_to_any_theme_and_only_when_the_quoted_phrase_holds_every_word_of_it():
    oil = ORDERS["I want to buy oil stocks. Buy the 3 most recommended companies."]["replies"][0]
    assert it.parse_intent(oil, "I want to buy oil stocks. Buy the 3 most recommended companies.").topic == "oil"
    assert it.parse_intent(oil, 'I want to buy "oil stocks". Buy the 3 most recommended companies.').topic == "oil stocks"
    assert it.parse_intent(oil, 'I want to buy oil stocks. Buy the 3 most recommended "companies".').topic == "oil"


# ---------------------------------------------------------------- T2 touches nothing else: a direct order, a ticker, a company's name

@pytest.mark.parametrize("typed, quoted", [
    ("Buy Microsoft", 'Buy "Microsoft"'),
    ("I want to buy MSFT", 'I want to buy "MSFT"'),
    ("sell MSFT", 'sell "MSFT"'),
    ("Buy $5,000 of MSFT", 'Buy $5,000 of "MSFT"'),
    ("I want to buy MSFT and TSLA and spend $6,000", 'I want to buy "MSFT" and "TSLA" and spend $6,000'),
])
def test_a_direct_order_is_read_the_same_with_its_names_in_quotes(typed, quoted):
    plain, with_quotes = it.parse_intent(ORDERS[typed]["replies"][0], typed), it.parse_intent(ORDERS[typed]["replies"][0], quoted)
    assert with_quotes.action == "order" and with_quotes.topic is None and with_quotes.model_dump(exclude={"request"}) == plain.model_dump(exclude={"request"})


def test_a_list_of_tickers_is_read_the_same_with_the_tickers_in_quotes():
    typed = "I want to buy MU, SNDK and AMD if the recommendations pass"
    plain, with_quotes = it.parse_intent(ORDERS[typed]["replies"][0], typed), it.parse_intent(ORDERS[typed]["replies"][0], 'I want to buy "MU", "SNDK" and "AMD" if the recommendations pass')
    assert (with_quotes.action, with_quotes.topic, with_quotes.symbols, with_quotes.companies) == ("buy", None, ["MU", "SNDK", "AMD"], []) and with_quotes.as_request() == ("MU SNDK AMD", "symbols")
    assert with_quotes.model_dump(exclude={"request"}) == plain.model_dump(exclude={"request"})


# ---------------------------------------------------------------- the theme in the lines that print it

@pytest.mark.parametrize("topic, what", [("AI stocks", "'AI stocks'"), ("AI Stocks.", "'AI Stocks.'"), ("home companies", "'home companies'"), ("tech shares", "'tech shares'"), ("AI", "'AI' companies"), ("stock market", "'stock market' companies"),
                                         ("oil and gas", "'oil and gas' companies")])
def test_a_theme_that_ends_in_the_noun_is_not_followed_by_it_again(topic, what):
    intent = it.TradeIntent(request="x", action="buy", topic=topic)
    assert intent.what() == what and intent.describe() == f"buy the most recommended {what}; spending as much as the account allows"


# ---------------------------------------------------------------- the live Auto session of Jacob's sentence

def test_the_real_auto_session_read_the_quoted_theme_asked_the_clerk_for_it_and_never_spent_more_than_the_20000_dollars():
    """`auto_ai_quoted_theme.json` is a capture of `session --name "Theme Check" --age 40 --risk moderate --cash 25000 --mode auto --yes --cleanup --save --prompt '<Jacob's sentence>'` (2026-11-16; NO fetch: the stored
    data; every agent real: the Account Manager, the Data Clerk, the Advisor, the Critic and the Trader over the real local Postgres; the simulated close of Friday 2026-11-13; a throwaway account that the run deleted).
    Only what is stable is pinned: the reading and the money. Which companies the Critic approved varies from run to run and is not."""
    data = json.loads((SESSIONS / "auto_ai_quoted_theme.json").read_text(encoding="utf-8"))
    assert data["request"] == JACOB and data["path"] == ["account_manager", "data_clerk", "advisor", "critic", "account_manager_decision", "trader"]
    assert {k: data["intent"][k] for k in ("action", "topic", "symbols", "companies", "count", "allocation", "budget", "orders")} == {"action": "buy", "topic": "AI stocks", "symbols": [], "companies": [], "count": 3, "allocation": "default", "budget": 20000, "orders": []}
    again = it.parse_intent(reply(JACOB), data["request"])
    assert (again.topic, again.count, again.budget, again.as_request()) == ("AI stocks", 3, Decimal("20000"), ("AI stocks", "phrase"))      # the real reply, read again by today's code: the same form
    order = wo.WorkOrder.model_validate(data["order"])
    assert order.status == "traded" and order.errors == [] and wo.verify_trades(order) == [] and order.trades
    spent = sum((money(t["fill"]["amount"]) for t in order.trades), Decimal("0.00"))
    before, after = money(order.trader_runs[-1]["portfolio_before"]["cash"]), money(order.trader_runs[-1]["portfolio_after"]["cash"])
    assert all(t["status"] == "filled" and t["side"] == "buy" and t["booked"] is True for t in order.trades) and {d["will_trade"] for d in order.trade_decisions if d["shares"]} == {"yes_auto"}
    assert before == Decimal("25000.00") and Decimal("0.00") < spent <= Decimal("20000.00") and before - spent == after      # never more than the budget; the cash after is the cash before less the fills, exactly
    assert sum((money(d["est_cost"]) for d in order.trade_decisions), Decimal("0.00")) <= Decimal("20000.00")
