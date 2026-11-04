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


@pytest.fixture(autouse=True)
def no_real_collections(request, monkeypatch):
    if "real_embeddings" in request.fixturenames:
        return
    monkeypatch.setattr("prometheus.rag.retriever.get_vector_store", lambda kind="news": EmptyStore())
