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
from pathlib import Path


from prometheus.agents.account_manager import intent as it
from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
from prometheus.common.money import money

SESSIONS = Path(__file__).parent / "fixtures" / "session_real"
THEMES = json.loads((SESSIONS / "theme_request_forms.json").read_text(encoding="utf-8"))["requests"]
JACOB = 'I want to buy the best "AI stocks" as a theme. Buy $20000 worth of the top 3'
NO_QUOTES = "I want to buy the best AI stocks. Buy $20000 worth of the top 3"
EV = "Buy the 3 most recommended EV stocks with $5,000"
FIVE_G = "Look at the best 5G companies"


def reply(typed: str) -> str:
    """The real model's one reply to this request."""
    return THEMES[typed]["replies"][0]


def read(typed: str, said: str | None = None):
    """The real reply to `typed`, checked against `said` (the sentence the user typed, which is `typed` unless a test reads the reply against another)."""
    return it.parse_intent(reply(typed), said or typed)


# ---------------------------------------------------------------- the real replies

def test_the_real_model_leaves_stocks_out_of_the_theme_so_it_is_a_word_of_two_letters():
    assert {typed: json.loads(reply(typed))["topic"] for typed in THEMES} == {JACOB: "AI", NO_QUOTES: "AI", EV: "EV", FIVE_G: "5G"} and all(len(THEMES[t]["replies"]) == 1 for t in THEMES)      # the model's own words, one call each


def test_the_phrase_rule_is_the_data_clerks_own():
    assert it.CLERK_PHRASE_CHARS in inspect.getsource(DataClerk.find_companies) and "len(text) > 60" in inspect.getsource(DataClerk.find_companies)
