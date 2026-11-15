"""Shared test setup.

A Data Clerk built without a store searches the REAL Chroma collections (data/chroma, the embedding model). Since design choices entry 40 every work order is swept over all three
collections, so a test that hands the clerk only the stores it cares about would reach into the real index for the others: slow, and dependent on what happens to be stored.
This fixture makes "no store given" mean an empty store, in every test; a test that wants a result passes its own store, as it always did. A test that asks for the
real embedding model (`real_embeddings`) builds a real index in a temporary folder and searches it through the default store, so it is left alone.
"""
import pytest


class EmptyStore:
    """Looks like a Chroma collection with nothing in it."""

    def similarity_search_with_score(self, query, k=4, filter=None):
        return []

    def get(self, ids=None, include=None, **options):
        return {"ids": [], "documents": [], "metadatas": []}


@pytest.fixture(autouse=True)
def no_real_collections(request, monkeypatch):
    if "real_embeddings" in request.fixturenames:
        return
    monkeypatch.setattr("prometheus.rag.retriever.get_vector_store", lambda kind="news": EmptyStore())


@pytest.fixture(autouse=True)
def market_hours_count(monkeypatch):
    """The Critic reads IGNORE_MARKET_HOURS from the environment, then from .env. A test must not inherit Jacob's .env: the closed market stays a blocker unless a test sets the flag itself."""
    monkeypatch.setenv("IGNORE_MARKET_HOURS", "false")


@pytest.fixture(autouse=True)
def paper_trading_on(monkeypatch):
    """The Trader reads PAPER_TRADING from the environment, then from .env. A test must not inherit Jacob's .env (the same reason as `market_hours_count`): paper trading is on unless a test turns it off itself."""
    monkeypatch.setenv("PAPER_TRADING", "true")


@pytest.fixture(autouse=True)
def live_clock_on(monkeypatch):
    """The desk's clock reads USE_LIVE_TRADING from the environment, then from .env (design choices entry 76). A test must not inherit Jacob's .env, which says false (the simulated clock and the stored closing price):
    the real clock is on unless a test turns it off itself."""
    monkeypatch.setenv("USE_LIVE_TRADING", "true")


@pytest.fixture(autouse=True)
def no_writes_to_the_real_ledger(monkeypatch):
    """No test may append to the project's real ledgers (data/ledger/: the Advisor's recommendations and the Critic's reviews, which a real run reads and writes). A test passes its own ledger in a temporary folder."""
    from prometheus.memory.ledger import Ledger
    from prometheus.rag.config import REPO_ROOT
    real, original = REPO_ROOT / "data" / "ledger", Ledger.append

    def guarded(self, record):
        if real in self.path.parents:
            raise AssertionError(f"a test tried to write to the real ledger {self.path}: give the Advisor or the Critic a ledger in a temporary folder")
        return original(self, record)

    monkeypatch.setattr(Ledger, "append", guarded)


@pytest.fixture(autouse=True)
def no_real_trader_model(monkeypatch):
    """A Trader built without a model builds the real one (OpenRouter, paid) the first time it needs it. In a test that must never happen: here `get_chat_model` fails the way a missing key does, so a Trader with
    no model of its own is a Trader whose model cannot start (the sweep still trades). A test of the model's own settings patches this name itself, with a fake that records what it is asked."""
    def refuse(*args, **kwargs):
        raise RuntimeError("OPENROUTER_API_KEY is not set (a test must not build a real model)")

    monkeypatch.setattr("prometheus.agents.trader.trader.get_chat_model", refuse)


@pytest.fixture(autouse=True)
def no_real_quotes_for_the_holdings(monkeypatch):
    """`AccountManager.decide_trades` asks the Data Clerk for the current price of the symbols the portfolio holds (`request_prices`, one quote order: Finnhub, paid). An Account Manager built with no Data Clerk would build the
    real one. In a test that must never happen: here such a manager's quote request fails the way a missing key does, which `decide_trades` meets by counting the holdings at their cost basis. A manager that is given a clerk
    (a test's own, over real recorded quotes) asks it as before."""
    from prometheus.agents.account_manager.account_manager import AccountManager
    original = AccountManager.request_prices

    def guarded(self, symbols):
        if self._clerk is None:
            raise RuntimeError("a test must not build a real Data Clerk for quotes: give the Account Manager a clerk")
        return original(self, symbols)

    monkeypatch.setattr(AccountManager, "request_prices", guarded)
