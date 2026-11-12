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
