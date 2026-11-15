"""A user session, end to end, with the REAL agents on REAL data (design choices entry 78).

Run:  PROMETHEUS_LIVE_TESTS=1 pytest tests/test_end_to_end_live.py -q -s

THIS TEST CALLS THE PAID APIs: OpenRouter for the Account Manager's, the Advisor's, the Critic's (GLM 5.3) and the Trader's models, and Finnhub for the Critic's numbers, a few cents and a few minutes in all. It
creates accounts in the local Postgres account book (the real one, `scripts/setup_postgres.sh`) and deletes every one it made; the three sample accounts are never touched. It runs with `--no-fetch`: the data are
the real data already stored (daily bars to 2026-11-13, news, press releases and filings of PG, MSFT, TSLA and GM), so nothing is downloaded and no file is written. It is skipped unless PROMETHEUS_LIVE_TESTS=1,
so that a plain `pytest` does not spend money (AGENTS.md).

NO STAND-IN AGENT AND NO INVENTED DATA. Every agent is the real class (`ss.build_parts`: Data Clerk, Advisor, Critic, Trader, Account Manager, the Postgres book), driven by the LangGraph desk exactly as
`./test_prometheus.sh --session` drives it. The clock is the simulated one (USE_LIVE_TRADING=false: the previous business day's close), which is what makes a run possible at night and at the weekend. What the
tests assert is what must hold whatever the models and the market say: the money adds up, the split is even to within one share when the sentence asks for one, nothing is bought above what the Account Manager
decided (the Account Manager approves each trade against the cash the book has at that moment and records each fill at once, design choices entry 81, and the Trader's single-trade output check refuses a fill above the decision), the decisions are saved, every node of the graph ran in order. They do not assert which companies are approved or how many shares: that is the market's and the models'
to decide, and the Account Manager's (an Auto session that asks for no split spends the funds round-robin, a share of each approved stock in turn, within the profile's limits: sizes are not asserted here).
"""
import os
import re
from datetime import date

import pytest

from prometheus.agents.account_manager.account_manager import AccountManager
from prometheus.application import session as ss
from prometheus.persistence.accounts import open_account_store

pytestmark = [pytest.mark.live, pytest.mark.skipif(os.environ.get("PROMETHEUS_LIVE_TESTS") != "1",
                                                   reason="calls the paid OpenRouter and Finnhub APIs: set PROMETHEUS_LIVE_TESTS=1 to run it")]

SENTENCE = ("I want to buy oil stocks. Buy the most amount of oil stocks shares from the 3 most recommended companies. "
            "Balance the purchased stocks evenly among the 3 recommended companies.")
STORED = "I want to buy PG, MSFT and GM if they are recommended. Buy the most shares of the three and split the money evenly among them."      # real tickers whose data are stored; it asks for recommendations (a named stock with no such word is a direct order: entry 98)
YEAR = (date(2026, 1, 1), date(2026, 12, 31))
NODES = ["account_manager", "data_clerk", "advisor", "critic", "account_manager_decision", "trader"]
CREATED: list[int] = []


@pytest.fixture(scope="module")
def book():
    return open_account_store()


@pytest.fixture(autouse=True)
def simulated_clock_and_clean_up(monkeypatch, book):
    monkeypatch.setenv("USE_LIVE_TRADING", "false")                  # the dev mode of .env: the previous business day's close, so a session runs at night and at the weekend
    yield
    for account_id in CREATED:
        try:
            book.delete_account(account_id)
        except ValueError:
            pass
    CREATED.clear()


def session(*argv, ask=None):
    """The session exactly as the script runs it (the real parts), with what it printed. The account it made is deleted after the test."""
    args = ss.build_parser().parse_args(argv)
    lines = []
    code = ss.main(list(argv), parts=ss.build_parts(args), ask=ask or (lambda prompt: pytest.fail(f"nobody was to be asked, and this was asked: {prompt!r}; what was printed so far: " + " | ".join(lines[-12:]))), out=lines.append)
    text = "\n".join(lines)
    made = re.search(r"^\s+account (\d+): ", text, re.M)
    if made and "ACCOUNT  created" in text:
        CREATED.append(int(made.group(1)))
    return code, text, int(made.group(1)) if made else None


USER = ["--name", "Ada Lovelace", "--age", "41", "--risk", "moderate", "--cash", "15000", "--mode", "auto", "--auto-min-score", "60", "--yes"]


def test_the_real_model_reads_the_sentence_as_a_theme_a_count_and_an_even_split():
    intent = AccountManager().understand_request(SENTENCE)
    assert intent.action == "buy" and intent.topic and "oil" in intent.topic.lower() and intent.count == 3 and intent.allocation == "even" and intent.budget is None
    assert intent.as_request() == (intent.topic, "phrase")


def test_the_real_model_reads_tickers_and_companies_as_such_and_asking_to_look_as_research():
    m = AccountManager()
    tickers = m.understand_request(STORED)
    assert tickers.symbols == ["PG", "MSFT", "GM"] and tickers.topic is None and tickers.allocation == "even" and tickers.count in (None, 3)      # "the three" is three tickers: the model may or may not give the count
    look = m.understand_request("Show me the best of Procter & Gamble and Microsoft, I am only looking.")
    assert look.action == "research" and look.companies == ["Procter & Gamble", "Microsoft"] and look.budget is None


def test_a_session_on_real_stored_data_runs_every_node_of_the_graph_and_the_account_book_agrees_with_what_was_printed(book):
    code, text, account_id = session("--prompt", STORED, *USER)
    assert code in (0, 1) and account_id, text                                                       # 0: it traded; 1: nothing was approved to trade, and it says why
    # the LangGraph desk ran, node by node, in order
    positions = [text.index(f".. {words}") for words in (ss.NODE_WORDS[n] for n in NODES[:5]) if f".. {words}" in text]
    assert positions == sorted(positions) and f".. {ss.NODE_WORDS['account_manager']}" in text and f".. {ss.NODE_WORDS['data_clerk']}" in text and f".. {ss.NODE_WORDS['advisor']}" in text
    assert re.search(r"understood: buy the (3 )?most recommended PG, MSFT, GM; the money spread evenly", text) and "RECOMMENDATIONS  asked for the top 3 of PG, MSFT, GM" in text      # the model may or may not give the count for "the three" tickers (see the test above): both are the same request
    after = book.list_portfolio(account_id)
    spent = sum(p.shares * p.cost_basis for p in after.positions)
    if code == 0:
        assert "SESSION  PASSED: the portfolio changed (" in text and after.positions
        assert abs((15000.0 - after.cash_avail) - spent) < 0.05 * max(1, len(after.positions))      # the cash the account lost is what the shares cost (to the cent of each fill)
        assert f"account  cash $15,000.00 -> ${after.cash_avail:,.2f}" in text and after.cash_avail >= 0     # since entry 81 the Trader holds no cash: the figure it prints after the run is the Account Manager's, which is the book's, to the cent
        assert book.number_trades(account_id, *YEAR) == len(after.positions) and book.number_approved_recommendations(account_id, *YEAR) >= len(after.positions)
        if len(after.positions) >= 2:                                                                 # the money is spread evenly, to within one share of the dearest
            costs = [p.shares * p.cost_basis for p in after.positions]
            assert max(costs) - min(costs) <= max(p.cost_basis for p in after.positions) * 1.01
    else:
        assert "the portfolio did NOT change. Why:" in text and after.positions == [] and after.cash_avail == 15000.0
    assert "ACCOUNT  after the session" in text


ASK_USER = ["--name", "Ada Lovelace", "--age", "41", "--risk", "moderate", "--cash", "15000", "--mode", "ask", "--yes"]       # Ask is the default: a person approves each trade (--yes answers only the understanding)


def person(answer, questions):
    """The user at the keyboard: answers every trade question with `answer`, and fails on any other question (--yes: nobody is asked to confirm what was understood)."""
    def ask(prompt):
        assert prompt.startswith(("Place this trade", "Sell these")), prompt
        questions.append(prompt)
        return answer
    return ask


def test_in_ask_mode_the_session_shows_the_proposal_and_places_only_what_the_user_approved(book):
    questions: list[str] = []
    code, text, account_id = session("--prompt", STORED, *ASK_USER, ask=person("y", questions))
    after = book.list_portfolio(account_id)
    assert code in (0, 1) and account_id, text
    if questions:
        assert "ASK MODE  nothing is traded without your yes" in text and text.index("ASK MODE") < text.index(f".. {ss.NODE_WORDS['account_manager_decision']}")      # the proposal first, then the questions, then the decision
        shown = {m.group(2): int(m.group(1)) for q in questions for m in [re.search(r"\((\d+) shares of (\w+),", q)] if m}
        assert after.positions and {p.symbol for p in after.positions} <= set(shown) and all(p.shares <= shown[p.symbol] for p in after.positions)     # never more than the size the user was shown
        assert "go on to the Trader" in text
    else:
        assert code == 1 and after.positions == [] and "nothing was approved to trade" in text


UNSAID = "I want to buy PG, MSFT and GM, the three most recommended."      # real tickers whose data are stored, and NO amount: in Ask mode the Account Manager asks how much (entry 85)


def test_in_ask_mode_a_request_that_does_not_say_how_much_is_asked_how_much_and_the_answer_sizes_what_is_proposed(book):
    """The real Account Manager's model words the question and reads the answer (a percentage of the account, for each company); code checks it and works out the shares; the user then approves each trade."""
    seen: list[str] = []

    def ask(prompt):
        seen.append(prompt)
        if prompt.startswith("Place this trade"):
            return "y"
        if "(question" in prompt:
            return "5% of my account in each company"
        pytest.fail(f"an unexpected question: {prompt!r}")

    code, text, account_id = session("--prompt", UNSAID, *ASK_USER, ask=ask)
    after = book.list_portfolio(account_id)
    how_much = [prompt for prompt in seen if "(question" in prompt]
    assert code in (0, 1) and account_id, text
    if how_much:
        assert "HOW MUCH?" in text and text.index("HOW MUCH?") < text.index("ASK MODE  nothing is traded without your yes")          # the question comes BEFORE the proposal of the trades
        said = re.search(r"How much: 5% of your account in each of the (\d) compan", text)
        assert said or "How much: 5% of your account in" in text, text
        for position in after.positions:
            assert position.shares * position.cost_basis <= 0.05 * 15000.0 + max(p.cost_basis for p in after.positions), position      # at most 5% of the account in each (one share of slack for the fill price)
    else:
        assert code == 1 and "nothing was approved to trade" in text                                                                        # the Critic approved none, so there was nothing to size


def test_a_stock_named_with_no_number_is_a_direct_order_in_ask_mode_the_user_is_asked_how_many_and_what_they_say_is_bought(book):
    """Design choices entry 98 (Jacob, 2026-11-15): "I want to buy MSFT" is a direct order, not a request to recommend: no research, no Advisor, no Critic. In Ask mode the real Account Manager asks how many, the real model
    words the question and reads the answer, code checks it, and the user is then shown the order and says yes."""
    seen: list[str] = []

    def ask(prompt):
        seen.append(prompt)
        if "(question" in prompt:
            return "3 shares"
        if prompt.startswith("Is this the order you meant"):
            return "y"
        pytest.fail(f"an unexpected question: {prompt!r}")

    code, text, account_id = session("--prompt", "I want to buy MSFT", *ASK_USER, ask=ask)
    after = book.list_portfolio(account_id)
    assert code == 0 and account_id and any("(question" in prompt for prompt in seen) and "DIRECT ORDER" in text and "the Advisor ranked" not in text, text
    assert [(p.symbol, p.shares) for p in after.positions] == [("MSFT", 3)] and "FILLED 3" in text


def test_in_ask_mode_a_no_to_every_trade_buys_nothing_and_the_trader_does_not_run(book):
    questions: list[str] = []
    code, text, account_id = session("--prompt", STORED, *ASK_USER, ask=person("n", questions))
    after = book.list_portfolio(account_id)
    assert code == 1 and after.positions == [] and after.cash_avail == 15000.0 and f".. {ss.NODE_WORDS['trader']}" not in text
    assert (not questions) or "nothing was approved to trade: the Trader will not run." in text


def test_saying_no_to_what_was_understood_stops_before_any_data_is_read_and_the_account_is_unchanged(book):
    code, text, account_id = session("--prompt", STORED, "--name", "Ada Lovelace", "--age", "41", "--risk", "moderate", "--cash", "15000", "--mode", "auto", ask=lambda prompt: "n")
    assert code == 2 and "Stopped. Nothing was bought or sold." in text and f".. {ss.NODE_WORDS['data_clerk']}" not in text
    after = book.list_portfolio(account_id)
    assert after.positions == [] and after.cash_avail == 15000.0 and book.number_trades(account_id, *YEAR) == 0


def test_asking_only_to_look_ends_after_the_advisor_and_buys_nothing(book):
    code, text, account_id = session("--prompt", "Show me how PG, MSFT and GM rank. I am only looking, I do not want to buy anything.", *USER)
    assert code == 0 and "SESSION  a look only" in text and f".. {ss.NODE_WORDS['critic']}" not in text and f".. {ss.NODE_WORDS['trader']}" not in text
    assert book.list_portfolio(account_id).positions == []


def test_a_request_the_words_cannot_support_stops_the_desk_at_the_first_node(book):
    code, text, account_id = session("--prompt", "hmm", *USER)
    assert code == 2 and f".. {ss.NODE_WORDS['data_clerk']}" not in text and book.list_portfolio(account_id).positions == []
